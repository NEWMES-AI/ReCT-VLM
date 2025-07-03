import os
from multiprocessing import Process
from vllm import LLM, SamplingParams
from vllm.utils import get_open_port
import argparse
import json
import csv
import time
import gc
import random
from time import sleep
from collections import defaultdict

# Define the list of abnormality classes
ABNORMALITY_CLASSES = [
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset of CT report analysis results with data parallel inference")
    
    parser.add_argument('--csv-path', type=str, default="/data2/CT-RATE/reports/train/train_reports.csv", help='Path to the CSV file')
    parser.add_argument('--output-file', type=str, default="ct_report_analysis_results_location.json", help='Output JSON file path')
    parser.add_argument('--output-dir', type=str, default="/home/compu/WHY/ct-chat/results/location", help='Directory to save individual JSON files')
    parser.add_argument('--output-csv', type=str, default="ct_report_answers_location.csv", help='Output CSV file for all answers')
    parser.add_argument('--questions-file', type=str, default="/home/compu/WHY/ct-chat/Dataset/location_template.json", help='Path to questions JSON file')
    ################ model parameters ################
    """
        For running the model in parallel on a single machine with 4 GPUs:
        --dp-size=1 (default)
        --tp-size=4 (default)
        --node-size=1 (default)
        --node-rank=0 (default)
    """
    # parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")

    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size") # Process 1 task at a time
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size") # Split one process across 4 GPUs
    parser.add_argument("--node-size", type=int, default=1, help="Total number of nodes") # Total number of nodes to use in the cluster
    parser.add_argument("--node-rank", type=int, default=0, help="Rank of the current node") # Index of the current running node (starts from 0)
    parser.add_argument("--master-addr", type=str, default="", help="Master node IP address") # IP address shared by all nodes in the cluster = central node's IP address / For single machine, it's automatically set to localhost internally with --master-port=get_open_port() -> no need to specify directly
    parser.add_argument("--master-port", type=int, default=0, help="Master node port") # Port number shared by all nodes in the cluster = central node's port number
    parser.add_argument("--enforce-eager", action='store_true', help="Enforce eager mode execution.") # Set model execution mode / eager mode = calculations performed immediately upon code execution, immediate forward execution (commonly used for inference, testing single prompts) / lazy mode = only plans execution and performs batch processing at optimal time, internal method used by vLLM (used for fast bulk inference)
    parser.add_argument("--trust-remote-code", action='store_true', help="Trust remote code.") # Security option when loading models from Huggingface / Determines whether to trust and execute external model code
    return parser.parse_args()

def main(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip, dp_master_port, GPUs_per_dp_rank, enforce_eager, trust_remote_code, args):
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # Define the prompt template
    prompt_template = """
    You are a radiologist.

    # Instructions — read carefully
    - Do NOT begin your response with any label such as "Answer:", "Response:", or similar.
    - Write in complete sentences, as a concise radiology interpretation suitable for a clinical note.
    
    Analyze the following CT scan report:

    Findings:
    {findings}

    Impression:
    {impression}

    Based on the CT scan report above, generate an answer caption for the following question:
    {question}

    Please provide your response as a radiologist would, offering a clear and concise interpretation with relevant medical details.
    """

    # Load questions from JSON file
    with open(args.questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Get the questions list from the location key
    location_questions = questions_data.get('location', [])
    if not location_questions:
        print(f"DP rank {global_dp_rank}: Error - No questions found under 'location' key in the questions file.")
        return
    
    # Create an LLM (load model only once)
    print(f"DP rank {global_dp_rank}: Loading model...")
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        max_model_len=4096,  # Context length limit
    )
    print(f"DP rank {global_dp_rank}: Model loaded successfully")

    # Load CT reports from CSV and organize by volumename
    volume_reports = defaultdict(list)
    with open(args.csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            volume_name = row.get('VolumeName', '')
            if volume_name:
                volume_reports[volume_name].append(row)
    
    print(f"DP rank {global_dp_rank}: Loaded {len(volume_reports)} volume groups from CSV")

    # Convert to list of volume groups for distribution
    all_volume_groups = list(volume_reports.items())
    if len(all_volume_groups) == 0:
        print(f"DP rank {global_dp_rank}: Error - No volume groups found in CSV file")
        return
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"rank_{global_dp_rank}_checkpoint.json")
    
    # Load checkpoint (if exists)
    processed_volumes = set()
    last_completed_idx = -1
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_volumes = set(checkpoint_data.get('processed_volumes', []))
                last_completed_idx = checkpoint_data.get('last_completed_idx', -1)
                print(f"DP rank {global_dp_rank}: Loaded checkpoint. {len(processed_volumes)} volumes already processed.")
        except Exception as e:
            print(f"DP rank {global_dp_rank}: Error loading checkpoint: {e}")
    
    # Improve DP distribution logic
    volumes_per_rank = max(1, len(all_volume_groups) // dp_size)
    start_idx = global_dp_rank * volumes_per_rank
    end_idx = min(start_idx + volumes_per_rank if global_dp_rank < dp_size - 1 else len(all_volume_groups), len(all_volume_groups))
    volumes_to_process = all_volume_groups[start_idx:end_idx]
    
    # Start from the last checkpoint (process from the interrupted point)
    if last_completed_idx >= 0:
        start_process_idx = last_completed_idx + 1 - start_idx  # Convert to relative index
        if start_process_idx > 0 and start_process_idx < len(volumes_to_process):
            volumes_to_process = volumes_to_process[start_process_idx:]
            print(f"DP rank {global_dp_rank}: Resuming from index {last_completed_idx+1} (relative index {start_process_idx})")
    
    print(f"DP rank {global_dp_rank} will process {len(volumes_to_process)} volume groups (from {start_idx} to {end_idx-1})")
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each volume group
    for vol_idx, (volume_name, reports) in enumerate(volumes_to_process):
        # Check if volume has already been processed
        if volume_name in processed_volumes:
            print(f"DP rank {global_dp_rank}: Skipping already processed volume {volume_name}")
            continue
            
        output_path = os.path.join(args.output_dir, f"volume_{volume_name}_rank_{global_dp_rank}.json")
        if os.path.exists(output_path):
            print(f"DP rank {global_dp_rank}: Skipping already processed volume {volume_name}")
            processed_volumes.add(volume_name)
            continue
        
        # Prepare questions for all reports in current volume
        prompts = []
        report_info = []  # Store report information for each prompt
        
        for report in reports:
            # Create combinations of all classes and random location questions for each report
            for abnormality in ABNORMALITY_CLASSES:
                # Randomly select one location question
                random_question = random.choice(location_questions)
                # Insert abnormality class name
                formatted_question = random_question.format(abnormality=abnormality)
                
                prompt = prompt_template.format(
                    findings=report.get('Findings_EN', ''),
                    impression=report.get('Impressions_EN', ''),
                    question=formatted_question
                )
                prompts.append(prompt)
                report_info.append({
                    'volumename': volume_name,
                    'abnormality': abnormality,
                    'question': formatted_question
                })
        
        # Set batch size (limit memory usage)
        batch_size = 10
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_info = report_info[i:i+batch_size]
            print(f"DP rank {global_dp_rank}: Processing volume {volume_name}, batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Generate
            outputs = llm.generate(batch_prompts, SamplingParams(temperature=0.2, top_p=0.95, max_tokens=256))
            
            # Save results
            for j, output in enumerate(outputs):
                prompt = output.prompt
                print("output: ", output)
                answer = output.outputs[0].text
                print("answer:", answer)
                info = batch_info[j]
                
                results.append({
                    "volumename": info['volumename'],
                    "findinng" : report.get('Findings_EN', ''),
                    "impression" : report.get('Impressions_EN', ''),
                    "question": info['question'],
                    "answer": answer
                })
            
            # Clear memory
            del outputs
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save results for current volume
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update checkpoint
        processed_volumes.add(volume_name)
        current_idx = start_idx + vol_idx if last_completed_idx < 0 else last_completed_idx + 1 + vol_idx
        
        # Save checkpoint
        checkpoint_data = {
            'processed_volumes': list(processed_volumes),
            'last_completed_idx': current_idx
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Clear results list after processing current volume
        results = []
        gc.collect()
        
        print(f"DP rank {global_dp_rank}: Completed volume {volume_name} ({vol_idx+1}/{len(volumes_to_process)}), updated checkpoint")
    
    print(f"DP rank {global_dp_rank}: All volumes processed successfully")

if __name__ == "__main__":

    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size


    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=main,
                       args=(args.model, dp_size, local_dp_rank,
                             global_dp_rank, dp_master_ip, dp_master_port,
                             tp_size, args.enforce_eager,
                             args.trust_remote_code, args))
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        # Remove timeout (wait indefinitely)
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
