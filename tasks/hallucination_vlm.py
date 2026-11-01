import torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

class HallucinationDataset:
    """
    Step 2: Dataset handling for LLaVA Hallucination task.
    """
    def __init__(self, hallucination_data, model_wrapper, correct_data=None, mean_cache_path=None):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device
        
        print("Processing hallucination examples...")
        self.hallucination_examples = self._process_dataset(hallucination_data, is_correct_dataset=False)
        
        self.mean_correct_cache = {}
        if mean_cache_path:
            print(f"Loading precomputed mean activations from {mean_cache_path}...")
            cache = torch.load(mean_cache_path, map_location=self.device)
            self.avg_visual = {int(k): v.to(self.device) for k, v in cache["avg_visual"].items()}
            self.avg_question = {int(k): v.to(self.device) for k, v in cache["avg_question"].items()}
            self.avg_answer = cache["avg_answer"].to(self.device)
            self.correct_examples = [1] # Dummy to pass the zero-check later
            print("Loaded mean activations successfully.")
        else:
            print("Processing correct examples for mean computation...")
            if correct_data:
                self.correct_examples = self._process_dataset(correct_data, is_correct_dataset=True)
            else:
                self.correct_examples = []
                
            print(f"Loaded {len(self.hallucination_examples)} hallucination examples.")
            print(f"Loaded {len(self.correct_examples)} correct examples.")
            
            if len(self.correct_examples) > 0:
                self._precompute_mean_correct()
            else:
                print("WARNING: No correct examples provided. mean_correct will be zeroes!")

    def _load_image(self, img_path_or_pil):
        if isinstance(img_path_or_pil, str):
            return Image.open(img_path_or_pil).convert("RGB")
        return img_path_or_pil
        
    def _process_dataset(self, data, is_correct_dataset=False):
        processed = []
        for i, item in tqdm(enumerate(data), total=len(data), desc="Processing examples"):
            img = self._load_image(item["image"])
            # Run a forward pass to acquire visual/question/answer sequence boundaries 
            # and (if evaluating correct examples) memory-copy the reference activations
            gen_token_id, logits, _ = self.wrapper.run_greedy(img, item["question"], answer_prefix=item.get("answer_prefix", ""), answer_prefix_token_ids=item.get("answer_prefix_token_ids"))
            
            # Populate token IDs if missing
            # Support both naming conventions:
            #   hallucinated_answer / correct_answer  (expected)
            #   model_output / gt_output              (actual JSON from extract_hallucinations.py)
            hall_answer = item.get("hallucinated_answer") or item.get("model_output")
            corr_answer = item.get("correct_answer") or item.get("gt_output")
            
            if "hallucinated_token_id" not in item and hall_answer:
                item["hallucinated_token_id"] = self.wrapper.get_correct_token_id(hall_answer)
            if "correct_token_id" not in item and corr_answer:
                item["correct_token_id"] = self.wrapper.get_correct_token_id(corr_answer)
                
            item["image_loaded"] = img # cache loaded image
            item["visual_positions"] = self.wrapper.visual_positions.copy()
            item["question_positions"] = self.wrapper.question_positions.copy()
            item["answer_position"] = self.wrapper.answer_position
            
            # If this belongs to the correct distribution layout, save its forward activations
            # so the model can construct the CD-T mean baselines
            if is_correct_dataset:
                item["activations"] = {k: v.cpu().clone() for k, v in self.wrapper.get_activations().items()}

            processed.append(item)
                    
        return processed

    def _precompute_mean_correct(self):
        """
        Step 4: Precompute mean_correct_s for every attention head.
        We align by offsets for visual, question regions, and -1 for answer.
        """
        print("Precomputing mean_correct activations...")
        num_layers = self.wrapper.num_layers
        num_heads = self.wrapper.num_heads
        head_dim = self.wrapper.head_dim
        
        # We'll store lists of activations aligned by their offsets
        self.visual_means = defaultdict(list)
        self.question_means = defaultdict(list)
        self.answer_means = []

        for item in tqdm(self.correct_examples, desc="Precomputing means"):
            acts = item["activations"] # layer -> (batch, seq, n_heads, d_head)
            vis_pos = item["visual_positions"]
            q_pos = item["question_positions"]
            ans_pos = item["answer_position"]
            
            v_len = len(vis_pos)
            q_len = len(q_pos)
            
            # Gather answer pos across all layers
            # Shape: (layers, n_heads, d_head)
            ans_act = torch.stack([acts[l][0, ans_pos, :, :] for l in range(num_layers)])
            self.answer_means.append(ans_act)
            
            for i in range(v_len):
                seq_idx = vis_pos[i]
                v_act = torch.stack([acts[l][0, seq_idx, :, :] for l in range(num_layers)])
                self.visual_means[i].append(v_act)
                
            for i in range(q_len):
                seq_idx = q_pos[i]
                q_act = torch.stack([acts[l][0, seq_idx, :, :] for l in range(num_layers)])
                self.question_means[i].append(q_act)

        # Average them
        self.avg_answer = torch.stack(self.answer_means).mean(dim=0).to(self.device).to(self.wrapper.W_U.dtype)  # (layers, heads, d)
        
        self.avg_visual = {}
        for i, lst in self.visual_means.items():
            self.avg_visual[i] = torch.stack(lst).mean(dim=0).to(self.device).to(self.wrapper.W_U.dtype)
            
        self.avg_question = {}
        for i, lst in self.question_means.items():
            self.avg_question[i] = torch.stack(lst).mean(dim=0).to(self.device).to(self.wrapper.W_U.dtype)

    def get_mean_correct(self, visual_positions, question_positions, answer_position):
        """
        Constructs the mean_correct tensor aligned to a specific example's sequence length.
        Returns: Dict mapping layer_idx -> Tensor (1, seq_len, num_heads, head_dim)
        """
        # If no correct examples, return zeros
        if len(self.correct_examples) == 0:
            seq_len = answer_position + 1
            z = torch.zeros((1, seq_len, self.wrapper.num_heads, self.wrapper.head_dim), 
                            dtype=self.wrapper.W_U.dtype, device=self.device)
            return {l: z for l in range(self.wrapper.num_layers)}
            
        num_layers = self.wrapper.num_layers
        seq_len = answer_position + 1
        
        # Reconstruct full sequence tensor from averaged components
        # Default to 0 for positions longer than what we have in the correct examples
        means = torch.zeros((num_layers, seq_len, self.wrapper.num_heads, self.wrapper.head_dim), 
                            dtype=self.wrapper.W_U.dtype, device=self.device)
                            
        for i, pos_idx in enumerate(visual_positions):
            if i in self.avg_visual:
                means[:, pos_idx, :, :] = self.avg_visual[i]
                
        for i, pos_idx in enumerate(question_positions):
            if i in self.avg_question:
                means[:, pos_idx, :, :] = self.avg_question[i]
                
        means[:, answer_position, :, :] = self.avg_answer
        
        return {l: means[l].unsqueeze(0) for l in range(num_layers)}
