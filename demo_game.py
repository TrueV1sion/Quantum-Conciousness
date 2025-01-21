import os
import torch
import pygame
import asyncio
import logging
from dataclasses import dataclass
from typing import List

# Set OpenMP environment variable to handle multiple runtime initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config import SystemConfig, UnifiedState, ProcessingDimension
from machine_learning import RLConfig, RLAgent
from processors import AdvancedQuantumProcessor

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Game state container"""
    player_position: tuple
    enemies: List[tuple]
    score: int
    difficulty: float
    player_patterns: torch.Tensor
    environment_state: torch.Tensor


class AdaptiveGame:
    """Proof of concept adaptive game using ML components"""
    
    def __init__(self):
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Adaptive Game Demo")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # Game state
        self.player_pos = [400, 300]
        self.enemies = [[100, 100], [700, 100], [400, 500]]
        self.score = 0
        self.difficulty = 1.0
        
        # Initialize ML components
        self.setup_ml_components()
        
        # Game metrics
        self.player_history = []
        self.difficulty_history = []
        self.adaptation_history = []
        
    def setup_ml_components(self):
        """Initialize machine learning components"""
        # Configure RL for pattern recognition
        self.rl_config = RLConfig(
            state_size=4,  # x, y, velocity_x, velocity_y
            action_size=4,  # up, down, left, right
            hidden_size=64,
            learning_rate=0.001
        )
        
        # Initialize pattern recognition
        self.pattern_recognizer = RLAgent(self.rl_config)
        
        # Initialize state processor with unified_dim first
        system_config = SystemConfig(
            unified_dim=128,  # Required parameter first
            quantum_dim=32,
            consciousness_dim=64
        )
        self.state_processor = AdvancedQuantumProcessor(system_config)
        
    def get_game_state(self) -> GameState:
        """Convert current game state to tensor format"""
        player_state = torch.tensor([
            self.player_pos[0] / 800,  # Normalize positions
            self.player_pos[1] / 600,
            self.score / 100,  # Normalize score
            self.difficulty
        ], dtype=torch.float32)
        
        # Create environment state
        env_state = torch.zeros((10, 10))  # Simple grid representation
        env_state[int(self.player_pos[1] / 60), 
                 int(self.player_pos[0] / 80)] = 1
        
        return GameState(
            player_position=tuple(self.player_pos),
            enemies=self.enemies,
            score=self.score,
            difficulty=self.difficulty,
            player_patterns=player_state,
            environment_state=env_state
        )
    
    def create_unified_state(self, game_state: GameState) -> UnifiedState:
        """Convert game state to unified state for quantum processing."""
        # Create feature vector
        features = [
            game_state.player_position[0] / 800,
            game_state.player_position[1] / 600
        ]
        for enemy in game_state.enemies:
            features.extend([enemy[0] / 800, enemy[1] / 600])
        features.extend([
            game_state.score / 100,
            game_state.difficulty
        ])
        
        # Ensure we have exactly quantum_dim features
        features = features[:32]  # Truncate if too many
        features = features + [0.0] * (32 - len(features))  # Pad if too few
        
        # Create quantum field with proper shape for processing
        # Shape: [batch_size, quantum_dim]
        quantum_field = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Create consciousness field
        consciousness_field = torch.zeros(1, 64)  # [batch, consciousness_dim]
        consciousness_field[0, 0] = game_state.score / 100
        consciousness_field[0, 1] = game_state.difficulty
        
        # Create coherence matrix
        coherence_matrix = torch.eye(32).unsqueeze(0)  # [batch, dim, dim]
        
        # Create resonance patterns
        resonance_patterns = {
            'player_movement': game_state.player_patterns.unsqueeze(0),
            'environment': game_state.environment_state.flatten().unsqueeze(0)
        }
        
        # Create dimensional signatures
        dimensional_signatures = {
            ProcessingDimension.PHYSICAL: float(game_state.difficulty),
            ProcessingDimension.QUANTUM: 0.5,
            ProcessingDimension.CONSCIOUSNESS: 0.5,
            ProcessingDimension.TEMPORAL: 0.5,
            ProcessingDimension.INFORMATIONAL: 0.5,
            ProcessingDimension.UNIFIED: 0.5,
            ProcessingDimension.TRANSCENDENT: 0.5
        }
        
        return UnifiedState(
            quantum_field=quantum_field,
            consciousness_field=consciousness_field,
            unified_field=None,  # Will be computed by processor
            coherence_matrix=coherence_matrix,
            resonance_patterns=resonance_patterns,
            dimensional_signatures=dimensional_signatures,
            temporal_phase=0.0,
            entanglement_map={'player': 1.0},
            wavelet_coefficients=None,
            metadata=None
        )
    
    async def update_game_state(self, state: GameState):
        """Update game based on ML predictions"""
        # Process state through pattern recognition
        _ = self.pattern_recognizer(state.player_patterns)
        
        # Create and process unified state
        unified_state = self.create_unified_state(state)
        processed = await self.state_processor.process_state(unified_state)
        
        # Adjust difficulty based on player performance
        if len(self.player_history) > 10:
            performance = sum(self.player_history[-10:]) / 10
            self.difficulty = max(1.0, min(2.0, 
                self.difficulty + (performance - 0.5) * 0.1))
        
        # Update enemy positions based on difficulty and processed state
        coherence = torch.mean(processed.coherence_matrix).item()
        difficulty_mod = self.difficulty * (1 + coherence)
        
        for enemy in self.enemies:
            # Move enemies toward player with difficulty-based speed
            dx = (self.player_pos[0] - enemy[0]) * 0.01 * difficulty_mod
            dy = (self.player_pos[1] - enemy[1]) * 0.01 * difficulty_mod
            enemy[0] += dx
            enemy[1] += dy
            
            # Keep enemies in bounds
            enemy[0] = max(0, min(800, enemy[0]))
            enemy[1] = max(0, min(600, enemy[1]))
    
    def render(self):
        """Render game state"""
        self.screen.fill(self.BLACK)
        
        # Draw player
        pygame.draw.circle(
            self.screen, 
            self.BLUE, 
            [int(self.player_pos[0]), int(self.player_pos[1])], 
            20
        )
        
        # Draw enemies
        for enemy in self.enemies:
            pygame.draw.circle(
                self.screen, 
                self.RED, 
                [int(enemy[0]), int(enemy[1])], 
                15
            )
        
        # Draw score and difficulty
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, self.WHITE)
        diff_text = font.render(
            f'Difficulty: {self.difficulty:.2f}', 
            True, 
            self.WHITE
        )
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(diff_text, (10, 50))
        
        pygame.display.flip()
    
    async def game_loop(self):
        """Main game loop"""
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # Handle player input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.player_pos[0] -= 5
            if keys[pygame.K_RIGHT]:
                self.player_pos[0] += 5
            if keys[pygame.K_UP]:
                self.player_pos[1] -= 5
            if keys[pygame.K_DOWN]:
                self.player_pos[1] += 5
                
            # Keep player in bounds
            self.player_pos[0] = max(0, min(800, self.player_pos[0]))
            self.player_pos[1] = max(0, min(600, self.player_pos[1]))
            
            # Update game state
            game_state = self.get_game_state()
            await self.update_game_state(game_state)
            
            # Check collisions and update score
            for enemy in self.enemies:
                dx = self.player_pos[0] - enemy[0]
                dy = self.player_pos[1] - enemy[1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < 35:  # Collision radius
                    self.score -= 10
                    self.player_history.append(0)
                else:
                    self.score += 1
                    self.player_history.append(1)
            
            # Render
            self.render()
            
            # Cap at 60 FPS
            self.clock.tick(60)
            
            # Allow other async operations
            await asyncio.sleep(0)
        
        pygame.quit()


async def main():
    game = AdaptiveGame()
    await game.game_loop()


if __name__ == "__main__":
    asyncio.run(main()) 