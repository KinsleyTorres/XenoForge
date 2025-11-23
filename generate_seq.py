import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

os.makedirs('predictions', exist_ok=True)

class ImprovedHybridChemProteinModel(nn.Module):
    """
    Pure PyTorch model - must match training architecture.
    """
    def __init__(self,
                 chem_encoder_name='seyonec/ChemBERTa-zinc-base-v1',
                 protein_decoder_name='hugohrban/progen2-small',
                 cross_attention_dim=1024,
                 freeze_decoder=False):
        super().__init__()

        self.freeze_decoder = freeze_decoder

        # Load encoders
        print(f"Loading chemical encoder: {chem_encoder_name}")
        self.chem_encoder = AutoModel.from_pretrained(chem_encoder_name)
        self.chem_encoder.eval()
        for param in self.chem_encoder.parameters():
            param.requires_grad = False

        print(f"Loading protein decoder: {protein_decoder_name}")
        self.protein_decoder = AutoModelForCausalLM.from_pretrained(
            protein_decoder_name,
            trust_remote_code=True
        )

        if freeze_decoder:
            self.protein_decoder.eval()
            for param in self.protein_decoder.parameters():
                param.requires_grad = False
        else:
            self.protein_decoder.train()
            for param in self.protein_decoder.parameters():
                param.requires_grad = True

        # Get dimensions
        self.chem_hidden_dim = self.chem_encoder.config.hidden_size
        if hasattr(self.protein_decoder.config, 'n_embd'):
            self.protein_hidden_dim = self.protein_decoder.config.n_embd
        else:
            self.protein_hidden_dim = self.protein_decoder.config.hidden_size
        self.vocab_size = self.protein_decoder.config.vocab_size

        # Projection layers
        self.encoder_projection = nn.Sequential(
            nn.Linear(self.chem_hidden_dim, cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attention_dim)
        )

        self.decoder_projection_in = nn.Sequential(
            nn.Linear(self.protein_hidden_dim, cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attention_dim)
        )

        # Cross-attention layers
        self.cross_attention_1 = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm_1 = nn.LayerNorm(cross_attention_dim)

        self.intermediate_projection = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim),
            nn.GELU(),
            nn.LayerNorm(cross_attention_dim)
        )

        self.cross_attention_2 = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm_2 = nn.LayerNorm(cross_attention_dim)

        # Project back
        self.decoder_projection_out = nn.Sequential(
            nn.Linear(cross_attention_dim, self.protein_hidden_dim),
            nn.LayerNorm(self.protein_hidden_dim)
        )

        self.dropout = nn.Dropout(0.1)

        # Output head (for frozen decoder only)
        if freeze_decoder:
            self.lm_head = nn.Linear(self.protein_hidden_dim, self.vocab_size, bias=False)

    def forward(self, chem_input_ids, chem_attention_mask,
                protein_input_ids, protein_attention_mask, training=False):
        """Forward pass."""

        # Encode chemistry (no gradients)
        with torch.no_grad():
            chem_outputs = self.chem_encoder(
                input_ids=chem_input_ids,
                attention_mask=chem_attention_mask
            )
            chem_hidden_states = chem_outputs.last_hidden_state

        # Encode protein
        if self.freeze_decoder:
            with torch.no_grad():
                decoder_outputs = self.protein_decoder(
                    input_ids=protein_input_ids,
                    attention_mask=protein_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                protein_hidden_states = decoder_outputs.hidden_states[-1]
        else:
            decoder_outputs = self.protein_decoder(
                input_ids=protein_input_ids,
                attention_mask=protein_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            protein_hidden_states = decoder_outputs.hidden_states[-1]

        # Project
        chem_projected = self.encoder_projection(chem_hidden_states)
        protein_projected = self.decoder_projection_in(protein_hidden_states)

        # First cross-attention
        cross_out_1, _ = self.cross_attention_1(
            query=protein_projected,
            key=chem_projected,
            value=chem_projected,
            key_padding_mask=(chem_attention_mask == 0),
            need_weights=False
        )

        if training:
            cross_out_1 = self.dropout(cross_out_1)

        protein_enhanced_1 = self.norm_1(protein_projected + cross_out_1)

        # Intermediate
        protein_intermediate = self.intermediate_projection(protein_enhanced_1)

        # Second cross-attention
        cross_out_2, _ = self.cross_attention_2(
            query=protein_intermediate,
            key=chem_projected,
            value=chem_projected,
            key_padding_mask=(chem_attention_mask == 0),
            need_weights=False
        )

        if training:
            cross_out_2 = self.dropout(cross_out_2)

        protein_enhanced_2 = self.norm_2(protein_intermediate + cross_out_2)

        # Project back
        protein_for_lm = self.decoder_projection_out(protein_enhanced_2)

        # Get logits
        if self.freeze_decoder:
            logits = self.lm_head(protein_for_lm)
        else:
            logits = self.protein_decoder.lm_head(protein_for_lm)

        return logits


# ============================================================================
# GENERATION FUNCTION
# ============================================================================

def generate_sequences(model, chem_tokenizer, protein_tokenizer, smiles_input,
                      num_samples=10, max_length=500, min_length=150,
                      temperature=0.75, use_eos=True, device='cuda'):
    """
    Generate protein sequences from SMILES input.
    """
    model.eval()

    print(f"\nGenerating {num_samples} sequences...")
    print(f"  SMILES: {smiles_input[:80]}{'...' if len(smiles_input) > 80 else ''}")
    print(f"  Temperature: {temperature}")
    print(f"  Length range: {min_length}-{max_length} aa")
    print(f"  Method: {'EOS-based' if use_eos else 'Heuristic'}\n")

    # Encode chemistry
    chem_encoded = chem_tokenizer(
        smiles_input,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    chem_input_ids = chem_encoded['input_ids'].to(device)
    chem_attention_mask = chem_encoded['attention_mask'].to(device)

    # Repeat for batch
    if num_samples > 1:
        chem_input_ids = chem_input_ids.repeat(num_samples, 1)
        chem_attention_mask = chem_attention_mask.repeat(num_samples, 1)

    # Get tokens
    m_token_id = protein_tokenizer.encode('M', add_special_tokens=False)[0]
    eos_token_id = protein_tokenizer.eos_token_id
    pad_token_id = protein_tokenizer.pad_token_id

    if eos_token_id is None:
        eos_token_id = pad_token_id
        print(f"Warning: No EOS token, using pad token")

    print(f"Generation tokens:")
    print(f"  Start (M): {m_token_id}")
    print(f"  EOS: {eos_token_id}")
    print(f"  PAD: {pad_token_id}\n")

    # Start with M
    generated_ids = torch.full((num_samples, 1), m_token_id, dtype=torch.long, device=device)

    # Track finished sequences
    finished = np.zeros(num_samples, dtype=bool)

    # Encode chemistry once
    with torch.no_grad():
        chem_outputs = model.chem_encoder(
            input_ids=chem_input_ids,
            attention_mask=chem_attention_mask
        )
        chem_hidden_states = chem_outputs.last_hidden_state
        chem_projected = model.encoder_projection(chem_hidden_states)

    # Track for heuristic stopping
    recent_tokens = [[] for _ in range(num_samples)]
    low_confidence = [0 for _ in range(num_samples)]

    # Track amino acids for constraints
    aa_counts = [{'M': 1} for _ in range(num_samples)]
    hydrophobic_aas = set('AILMFVPW')
    charged_aas = set('DEKR')
    polar_aas = set('STNQ')
    helix_breakers = set('PG')

    print("Generating...")
    for step in range(max_length - 1):
        if finished.all():
            print(f"  All sequences finished at step {step + 1}")
            break

        current_length = step + 1

        # Create attention mask
        protein_attention_mask = torch.ones_like(generated_ids)

        # Forward pass
        with torch.no_grad():
            # Get protein hidden states
            decoder_outputs = model.protein_decoder(
                input_ids=generated_ids,
                attention_mask=protein_attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            protein_hidden_states = decoder_outputs.hidden_states[-1]

            # Project
            protein_projected = model.decoder_projection_in(protein_hidden_states)

            # Cross-attention 1
            cross_out_1, _ = model.cross_attention_1(
                query=protein_projected,
                key=chem_projected,
                value=chem_projected,
                key_padding_mask=(chem_attention_mask == 0),
                need_weights=False
            )
            protein_enhanced_1 = model.norm_1(protein_projected + cross_out_1)

            # Intermediate
            protein_intermediate = model.intermediate_projection(protein_enhanced_1)

            # Cross-attention 2
            cross_out_2, _ = model.cross_attention_2(
                query=protein_intermediate,
                key=chem_projected,
                value=chem_projected,
                key_padding_mask=(chem_attention_mask == 0),
                need_weights=False
            )
            protein_enhanced_2 = model.norm_2(protein_intermediate + cross_out_2)

            # Project back
            protein_for_lm = model.decoder_projection_out(protein_enhanced_2)

            # Get logits
            if model.freeze_decoder:
                logits = model.lm_head(protein_for_lm)
            else:
                logits = model.protein_decoder.lm_head(protein_for_lm)

        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)

        # Convert to numpy for manipulation
        next_token_logits_np = next_token_logits.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Apply constraints per sequence
        for batch_idx in range(num_samples):
            if finished[batch_idx]:
                # Force padding
                next_token_logits_np[batch_idx, :] = -1000.0
                next_token_logits_np[batch_idx, pad_token_id] = 100.0
                continue

            # Heuristic stopping
            if not use_eos and current_length >= min_length:
                # Check repetition
                if len(recent_tokens[batch_idx]) >= 10:
                    last_10 = recent_tokens[batch_idx][-10:]
                    if len(set(last_10)) <= 3:
                        finished[batch_idx] = True
                        print(f"  Seq {batch_idx + 1} stopped at {current_length} aa (repetition)")
                        continue

                # Check confidence
                max_prob = probs_np[batch_idx].max()
                if max_prob < 0.3:
                    low_confidence[batch_idx] += 1
                    if low_confidence[batch_idx] >= 5:
                        finished[batch_idx] = True
                        print(f"  Seq {batch_idx + 1} stopped at {current_length} aa (low confidence)")
                        continue
                else:
                    low_confidence[batch_idx] = 0

                # Random stopping
                if current_length >= min_length + 50:
                    stop_prob = (current_length - min_length - 50) / (max_length - min_length - 50)
                    stop_prob = min(stop_prob * 0.03, 0.08)
                    if np.random.random() < stop_prob:
                        finished[batch_idx] = True
                        print(f"  Seq {batch_idx + 1} stopped at {current_length} aa (random)")
                        continue

            # EOS manipulation
            if use_eos:
                if current_length < min_length:
                    next_token_logits_np[batch_idx, eos_token_id] = -1000.0
                else:
                    progress = (current_length - min_length) / (max_length - min_length)
                    eos_boost = progress * 5.0
                    next_token_logits_np[batch_idx, eos_token_id] += eos_boost

            # Composition constraints
            counts = aa_counts[batch_idx]
            total = current_length

            hydro_ratio = sum(counts.get(aa, 0) for aa in hydrophobic_aas) / total
            charged_ratio = sum(counts.get(aa, 0) for aa in charged_aas) / total

            if hydro_ratio < 0.30:
                for aa in hydrophobic_aas:
                    aa_token = protein_tokenizer.encode(aa, add_special_tokens=False)[0]
                    next_token_logits_np[batch_idx, aa_token] += 2.0
            elif hydro_ratio > 0.50:
                for aa in hydrophobic_aas:
                    aa_token = protein_tokenizer.encode(aa, add_special_tokens=False)[0]
                    next_token_logits_np[batch_idx, aa_token] -= 2.0

            if charged_ratio < 0.15:
                for aa in charged_aas:
                    aa_token = protein_tokenizer.encode(aa, add_special_tokens=False)[0]
                    next_token_logits_np[batch_idx, aa_token] += 1.5
            elif charged_ratio > 0.30:
                for aa in charged_aas:
                    aa_token = protein_tokenizer.encode(aa, add_special_tokens=False)[0]
                    next_token_logits_np[batch_idx, aa_token] -= 1.5

        # Sample next tokens
        next_token_logits_torch = torch.from_numpy(next_token_logits_np).to(device)
        probs_sampling = torch.softmax(next_token_logits_torch, dim=-1)
        next_tokens = torch.multinomial(probs_sampling, num_samples=1)

        # Update tracking
        next_tokens_np = next_tokens.cpu().numpy()
        for batch_idx in range(num_samples):
            if finished[batch_idx]:
                continue

            token = next_tokens_np[batch_idx, 0]

            # Check EOS
            if use_eos and token == eos_token_id:
                finished[batch_idx] = True
                print(f"  Seq {batch_idx + 1} stopped at {current_length + 1} aa (EOS)")
            else:
                # Update counts
                next_aa = protein_tokenizer.decode([token]).strip()
                if next_aa:
                    aa_counts[batch_idx][next_aa] = aa_counts[batch_idx].get(next_aa, 0) + 1
                    recent_tokens[batch_idx].append(token)
                    if len(recent_tokens[batch_idx]) > 20:
                        recent_tokens[batch_idx].pop(0)

        # Append
        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

        # Progress
        if (step + 1) % 100 == 0:
            num_finished = finished.sum()
            print(f"  Step {step + 1}/{max_length}: {num_finished}/{num_samples} finished")

    print(f"\nGeneration complete!\n")

    return generated_ids, finished


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_sequence_quality(sequence):
    """Analyze protein sequence quality."""
    hydrophobic = set('AILMFVPW')
    polar = set('STNQ')
    charged = set('DEKR')

    aa_counts = Counter(sequence)

    has_repeats = any(sequence.count(aa * 5) > 0 for aa in 'ACDEFGHIKLMNPQRSTVWY')
    has_long_repeats = any(sequence.count(aa * 10) > 0 for aa in 'ACDEFGHIKLMNPQRSTVWY')

    low_complexity = False
    window = 20
    for i in range(len(sequence) - window):
        window_seq = sequence[i:i+window]
        most_common_aa, count = Counter(window_seq).most_common(1)[0]
        if count / window > 0.4:
            low_complexity = True
            break

    diversity = len(aa_counts) / 20.0

    total = len(sequence)
    hydrophobic_ratio = sum(aa_counts[aa] for aa in hydrophobic) / total
    polar_ratio = sum(aa_counts[aa] for aa in polar) / total
    charged_ratio = sum(aa_counts[aa] for aa in charged) / total

    score = 100.0

    if has_long_repeats:
        score -= 50
    elif has_repeats:
        score -= 25

    if low_complexity:
        score -= 30

    score += diversity * 30

    if hydrophobic_ratio < 0.25 or hydrophobic_ratio > 0.50:
        score -= 20
    if charged_ratio < 0.12 or charged_ratio > 0.30:
        score -= 15
    if polar_ratio < 0.12 or polar_ratio > 0.35:
        score -= 15

    proline_ratio = aa_counts.get('P', 0) / total
    glycine_ratio = aa_counts.get('G', 0) / total

    if proline_ratio > 0.12:
        score -= 20
    if glycine_ratio > 0.12:
        score -= 20

    if total < 150:
        score -= 20
    elif total > 500:
        score -= 15

    if sequence and sequence[0] == 'M':
        score += 10

    return {
        'score': max(0, score),
        'has_repeats': has_repeats,
        'has_long_repeats': has_long_repeats,
        'low_complexity': low_complexity,
        'diversity': diversity,
        'length': total,
        'hydrophobic_ratio': hydrophobic_ratio,
        'polar_ratio': polar_ratio,
        'charged_ratio': charged_ratio,
        'proline_ratio': proline_ratio,
        'glycine_ratio': glycine_ratio
    }


def preprocess_smiles(smiles_string):
    """Preprocess SMILES string."""
    if '</s></s>' in smiles_string:
        parts = smiles_string.split('</s></s>')
        substrates = parts[0].strip()
        products = parts[1].strip() if len(parts) > 1 else ""
    else:
        substrates = smiles_string
        products = ""

    return {
        'substrates': substrates,
        'products': products,
        'full_reaction': smiles_string,
        'reaction_arrow': f"{substrates}>>{products}"
    }


# ============================================================================
# MAIN LOADING AND PREDICTION FUNCTIONS
# ============================================================================

def load_model_from_checkpoint(checkpoint_path,
                               decoder_choice='progen2-small',
                               freeze_decoder=False):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        decoder_choice: 'progen2-small' or 'progen2-medium'
        freeze_decoder: Must match training setting

    Returns:
        model: Loaded model
        tokenizers: Dict with chem and protein tokenizers
        device: Device model is on
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL FROM CHECKPOINT")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Decoder: {decoder_choice}")
    print(f"  Freeze decoder: {freeze_decoder}")
    print(f"{'='*60}\n")

    # Model configs
    configs = {
        'progen2-small': ('hugohrban/progen2-small', 1024),
        'progen2-medium': ('huggingface/progen2-medium', 1280),
    }

    if decoder_choice not in configs:
        raise ValueError(f"Unknown decoder: {decoder_choice}")

    model_name, cross_attn_dim = configs[decoder_choice]

    # Create model architecture
    print("Creating model architecture...")
    model = ImprovedHybridChemProteinModel(
        chem_encoder_name='seyonec/ChemBERTa-zinc-base-v1',
        protein_decoder_name=model_name,
        cross_attention_dim=cross_attn_dim,
        freeze_decoder=freeze_decoder
    )

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint['loss']:.4f}")

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = model.to(device)
    model.eval()

    # Load tokenizers
    print("\nLoading tokenizers...")
    chem_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    protein_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if protein_tokenizer.pad_token is None:
        if protein_tokenizer.eos_token is not None:
            protein_tokenizer.pad_token = protein_tokenizer.eos_token
        else:
            protein_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"  Chemical tokenizer: seyonec/ChemBERTa-zinc-base-v1")
    print(f"  Protein tokenizer: {model_name}")
    print(f"  EOS token: {protein_tokenizer.eos_token}")
    print(f"  EOS token ID: {protein_tokenizer.eos_token_id}")

    print(f"\n{'='*60}")
    print("MODEL READY FOR PREDICTION")
    print(f"{'='*60}\n")

    tokenizers = {
        'chem': chem_tokenizer,
        'protein': protein_tokenizer
    }

    return model, tokenizers, device


def predict_sequences(model, tokenizers, device, smiles_input,
                     num_samples=10, temperature=0.75,
                     max_length=500, min_length=150,
                     use_eos=True, min_quality_score=65,
                     smiles_format='full'):
    """
    Generate and analyze protein sequences.

    Args:
        model: Loaded model
        tokenizers: Dict with chem and protein tokenizers
        device: Device to run on
        smiles_input: SMILES reaction string
        num_samples: Number of sequences to generate
        temperature: Sampling temperature
        max_length: Maximum sequence length
        min_length: Minimum before allowing stop
        use_eos: Use EOS-based stopping
        min_quality_score: Minimum quality to pass filter
        smiles_format: Which part of SMILES to use

    Returns:
        sequences: List of sequence dicts sorted by quality
    """
    print(f"\n{'='*60}")
    print("PROTEIN SEQUENCE PREDICTION")
    print(f"{'='*60}")

    # Preprocess SMILES
    processed = preprocess_smiles(smiles_input)

    if smiles_format == 'full':
        smiles_for_encoding = processed['full_reaction']
    elif smiles_format == 'substrates':
        smiles_for_encoding = processed['substrates']
    elif smiles_format == 'products':
        smiles_for_encoding = processed['products']
    elif smiles_format == 'reaction_arrow':
        smiles_for_encoding = processed['reaction_arrow']
    else:
        smiles_for_encoding = processed['full_reaction']

    print(f"Input:")
    print(f"  Substrates: {processed['substrates'][:60]}...")
    print(f"  Products: {processed['products'][:60]}...")
    print(f"  Format: {smiles_format}")
    print(f"\nSettings:")
    print(f"  Num samples: {num_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Length: {min_length}-{max_length} aa")
    print(f"  Stopping: {'EOS' if use_eos else 'Heuristic'}")
    print(f"{'='*60}")

    # Generate
    generated_ids, finished = generate_sequences(
        model=model,
        chem_tokenizer=tokenizers['chem'],
        protein_tokenizer=tokenizers['protein'],
        smiles_input=smiles_for_encoding,
        num_samples=num_samples,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        use_eos=use_eos,
        device=device
    )

    # Decode and analyze
    print("Analyzing sequences...")
    sequences = []
    eos_token_id = tokenizers['protein'].eos_token_id

    for i in range(num_samples):
        seq_ids = generated_ids[i].cpu().numpy()

        # Truncate at EOS
        if use_eos and eos_token_id is not None and eos_token_id in seq_ids:
            eos_pos = list(seq_ids).index(eos_token_id)
            seq_ids = seq_ids[:eos_pos]

        # Decode
        sequence = tokenizers['protein'].decode(seq_ids, skip_special_tokens=True)
        sequence = sequence.replace(' ', '')

        # Force M start
        if len(sequence) > 0 and sequence[0] != 'M':
            sequence = 'M' + sequence[1:]

        # Analyze
        quality = analyze_sequence_quality(sequence)

        sequences.append({
            'id': i + 1,
            'sequence': sequence,
            'length': quality['length'],
            'quality_score': quality['score'],
            'has_repeats': quality['has_repeats'],
            'has_long_repeats': quality['has_long_repeats'],
            'low_complexity': quality['low_complexity'],
            'diversity': quality['diversity'],
            'hydrophobic_ratio': quality['hydrophobic_ratio'],
            'polar_ratio': quality['polar_ratio'],
            'charged_ratio': quality['charged_ratio'],
            'terminated_naturally': (use_eos and eos_token_id is not None and
                                    eos_token_id in generated_ids[i].cpu().numpy())
        })

    # Filter and sort
    high_quality = [s for s in sequences if s['quality_score'] >= min_quality_score]

    if not high_quality:
        print(f"\n⚠️  No sequences passed threshold ({min_quality_score})")
        print(f"   Using all sequences")
        high_quality = sequences

    high_quality.sort(key=lambda x: x['quality_score'], reverse=True)

    # Statistics
    lengths = [s['length'] for s in sequences]
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Length Statistics:")
    print(f"  Range: {min(lengths)}-{max(lengths)} aa")
    print(f"  Mean: {np.mean(lengths):.1f} aa")
    print(f"  Median: {np.median(lengths):.1f} aa")
    print(f"  Std Dev: {np.std(lengths):.1f} aa")

    if use_eos:
        natural_stops = sum(1 for s in sequences if s['terminated_naturally'])
        print(f"  Natural termination (EOS): {natural_stops}/{num_samples}")

    print(f"\nQuality:")
    print(f"  Generated: {num_samples}")
    print(f"  Passed filter (>={min_quality_score}): {len(high_quality)}")

    # Show top sequences
    print(f"\n{'='*60}")
    print("TOP 5 SEQUENCES")
    print(f"{'='*60}\n")

    for i, seq_info in enumerate(high_quality[:5]):
        term = "EOS" if seq_info.get('terminated_naturally', False) else "HEURISTIC"
        print(f"Sequence {seq_info['id']} [{term}] (Quality: {seq_info['quality_score']:.1f})")
        print(f"  Length: {seq_info['length']} aa")
        print(f"  Composition:")
        print(f"    Hydrophobic: {seq_info['hydrophobic_ratio']:.1%}")
        print(f"    Charged: {seq_info['charged_ratio']:.1%}")
        print(f"    Polar: {seq_info['polar_ratio']:.1%}")
        print(f"  Quality:")
        print(f"    Diversity: {seq_info['diversity']:.2f}")
        print(f"    Repeats: {seq_info['has_repeats']}")
        print(f"    Low complexity: {seq_info['low_complexity']}")
        print(f"  Preview: {seq_info['sequence'][:70]}...")
        print()

    return high_quality


def save_sequences_csv(sequences, output_path='predictions/sequences.csv'):
    """Save sequences to CSV."""
    import pandas as pd
    df = pd.DataFrame(sequences)
    df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")


def save_sequences_fasta(sequences, output_path='predictions/sequences.fasta'):
    """Save sequences to FASTA."""
    with open(output_path, 'w') as f:
        for seq_info in sequences:
            header = (f">seq_{seq_info['id']}_len{seq_info['length']}_"
                     f"q{seq_info['quality_score']:.0f}")
            f.write(f"{header}\n")

            seq = seq_info['sequence']
            for i in range(0, len(seq), 80):
                f.write(f"{seq[i:i+80]}\n")

    print(f"Saved FASTA: {output_path}")

#example usage

def generate_new_seqs(checkpoint_path, smiles_formatted, decoder_choice='progen2-small', freeze_decoder=False):
    print("\n" + "="*60)
    print("PROTEIN SEQUENCE GENERATION FROM CHECKPOINT")
    print("="*60 + "\n")


    model, tokenizers, device = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        decoder_choice=decoder_choice,
        freeze_decoder=freeze_decoder
    )

    # Example 1: Simple ester hydrolysis
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Ester Hydrolysis")
    print("="*60)


    sequences_1 = predict_sequences(
        model=model,
        tokenizers=tokenizers,
        device=device,
        smiles_input=smiles_formatted,
        num_samples=2,
        temperature=0.75,
        max_length=800,
        min_length=150,
        use_eos=True,
        min_quality_score=65,
        smiles_format='full'
    )

    # Save results
    save_sequences_csv(sequences_1, 'predictions/example1_sequences.csv')
    save_sequences_fasta(sequences_1, 'predictions/example1_sequences.fasta')

#path_1 = r"C:\Users\admin\PycharmProjects\XenoForge\ester_hydrolase_model\pytorch_eos_progen2-small_epoch_12.pt"

#smiles_1 = "O.OC[C@H]1O[C@@H](Oc2cc(O)cc(O)c2C(=O)CCc3ccc(O)cc3)[C@H](O)[C@@H](O)[C@@H]1O</s></s>OC[C@H]1O[C@@H](Oc2cc(O)cc(O)c2C(=O)CCc3ccc(O)cc3)[C@H](O)[C@@H](O)[C@@H]1O"

#generate_new_seqs(path_1, smiles_1)