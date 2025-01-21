import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, List
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader

from quantum_llm import QuantumLLM, QuantumLLMConfig
from consciousness_model import SystemState
from pathways import PathwayMode

@dataclass
class TrainingConfig:
    """Configuration for quantum LLM training."""
    batch_size: int = 8
    num_epochs: int = 10
    classical_lr: float = 1e-5
    quantum_lr: float = 1e-4
    consciousness_lr: float = 1e-4
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    use_wandb: bool = True

class QuantumLLMTrainer:
    """Trainer for Quantum-enhanced Language Models."""
    
    def __init__(
        self,
        model: QuantumLLM,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizers
        self.classical_optimizer = optim.AdamW(
            self.model.base_model.parameters(),
            lr=self.config.classical_lr
        )
        
        self.quantum_optimizer = optim.AdamW(
            [p for n, p in self.model.named_parameters() if "quantum" in n],
            lr=self.config.quantum_lr
        )
        
        self.consciousness_optimizer = optim.AdamW(
            [p for n, p in self.model.named_parameters() if "consciousness" in n],
            lr=self.config.consciousness_lr
        )
        
        # Learning rate schedulers
        self.classical_scheduler = self._create_scheduler(self.classical_optimizer)
        self.quantum_scheduler = self._create_scheduler(self.quantum_optimizer)
        self.consciousness_scheduler = self._create_scheduler(self.consciousness_optimizer)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        if self.config.use_wandb:
            wandb.init(project="quantum-llm", config=self.config.__dict__)
    
    def _create_scheduler(self, optimizer: optim.Optimizer):
        """Create a learning rate scheduler with warmup."""
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=len(self.train_dataloader) * self.config.num_epochs,
            pct_start=0.1
        )
    
    def train(self):
        """Train the quantum-enhanced language model."""
        self.model.train()
        total_steps = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            consciousness_state = None
            
            with tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
                for step, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        consciousness_state=consciousness_state
                    )
                    
                    # Calculate losses
                    classical_loss = self._compute_language_loss(
                        outputs["logits"],
                        batch["labels"]
                    )
                    quantum_loss = self._compute_quantum_loss(outputs["quantum_state"])
                    consciousness_loss = self._compute_consciousness_loss(
                        outputs["consciousness_state"]
                    )
                    
                    # Combined loss
                    total_loss = (
                        classical_loss +
                        0.1 * quantum_loss +
                        0.1 * consciousness_loss
                    ) / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Update consciousness state
                    consciousness_state = outputs["consciousness_state"].detach()
                    
                    # Gradient accumulation
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Clip gradients
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        
                        # Optimize
                        self.classical_optimizer.step()
                        self.quantum_optimizer.step()
                        self.consciousness_optimizer.step()
                        
                        # Schedule learning rates
                        self.classical_scheduler.step()
                        self.quantum_scheduler.step()
                        self.consciousness_scheduler.step()
                        
                        # Zero gradients
                        self.classical_optimizer.zero_grad()
                        self.quantum_optimizer.zero_grad()
                        self.consciousness_optimizer.zero_grad()
                        
                        total_steps += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                        'classical_loss': classical_loss.item(),
                        'quantum_loss': quantum_loss.item(),
                        'consciousness_loss': consciousness_loss.item()
                    })
                    
                    # Log metrics
                    if self.config.use_wandb and total_steps % self.config.logging_steps == 0:
                        wandb.log({
                            'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                            'classical_loss': classical_loss.item(),
                            'quantum_loss': quantum_loss.item(),
                            'consciousness_loss': consciousness_loss.item(),
                            'learning_rate': self.classical_scheduler.get_last_lr()[0]
                        })
                    
                    # Evaluation
                    if self.eval_dataloader is not None and total_steps % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        if self.config.use_wandb:
                            wandb.log(eval_metrics)
                        self.model.train()
    
    def _compute_language_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the language modeling loss."""
        return nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
    
    def _compute_quantum_loss(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Compute quantum coherence loss."""
        # Implement quantum state optimization objectives
        coherence = torch.abs(quantum_state).mean()
        entropy = -(quantum_state * torch.log(quantum_state + 1e-10)).mean()
        return -coherence + 0.1 * entropy
    
    def _compute_consciousness_loss(self, consciousness_state: SystemState) -> torch.Tensor:
        """Compute consciousness optimization loss."""
        # Implement consciousness state optimization objectives
        return consciousness_state.coherence_loss()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        self.model.eval()
        total_loss = 0
        consciousness_state = None
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    consciousness_state=consciousness_state
                )
                
                loss = self._compute_language_loss(outputs["logits"], batch["labels"])
                total_loss += loss.item()
                consciousness_state = outputs["consciousness_state"]
        
        return {
            'eval_loss': total_loss / len(self.eval_dataloader),
            'eval_perplexity': torch.exp(torch.tensor(total_loss / len(self.eval_dataloader)))
        } 