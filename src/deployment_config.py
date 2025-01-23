"""
Production deployment configuration and model optimization strategies.
Handles ASGI server settings, model caching, and parallelization.
"""

import logging
import multiprocessing
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import uvicorn
from fastapi import FastAPI
from pydantic import BaseSettings
from redis import Redis
from celery import Celery

class DeploymentConfig(BaseSettings):
    """Production deployment settings with environment variable support."""
    
    # ASGI Server Configuration
    workers: int = min(multiprocessing.cpu_count() + 1, 8)
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    reload: bool = False
    
    # GPU and Model Configuration
    model_parallel: bool = False
    model_parallel_size: int = 1
    use_gpu_cache: bool = True
    gpu_memory_fraction: float = 0.9
    
    # Redis Queue Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Model Warm-up
    warm_up_batches: int = 5
    batch_size: int = 32
    
    class Config:
        env_prefix = "QC_"  # Environment variables prefix: QC_WORKERS, QC_HOST, etc.

class ModelCache:
    """Manages model caching and warm-up strategies."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection for model caching."""
        try:
            self.redis_client = Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db
            )
            self.redis_client.ping()  # Test connection
            logging.info("Redis cache connection established")
        except Exception as e:
            logging.warning(f"Redis cache initialization failed: {e}")
            self.redis_client = None
    
    def warm_up_model(self, model: torch.nn.Module) -> None:
        """Perform model warm-up to optimize GPU memory allocation."""
        if not self.config.use_gpu_cache or not torch.cuda.is_available():
            return
            
        try:
            logging.info("Starting model warm-up...")
            model.eval()
            
            with torch.no_grad():
                for _ in range(self.config.warm_up_batches):
                    # Generate dummy input of appropriate size
                    dummy_input = torch.randn(
                        self.config.batch_size,
                        512,  # Typical sequence length
                        device=next(model.parameters()).device
                    )
                    # Run forward pass
                    _ = model(dummy_input)
                    
            torch.cuda.empty_cache()
            logging.info("Model warm-up completed")
            
        except Exception as e:
            logging.error(f"Model warm-up failed: {e}")

def setup_model_parallel(
    model: torch.nn.Module,
    config: DeploymentConfig
) -> torch.nn.Module:
    """Configure model parallelization if enabled."""
    if not config.model_parallel or not torch.cuda.is_available():
        return model
        
    try:
        # Initialize distributed environment
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        
        # Wrap model with DistributedDataParallel
        model = model.cuda()
        model = DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank()
        )
        logging.info(
            f"Model parallel setup complete on {dist.get_world_size()} devices"
        )
        return model
        
    except Exception as e:
        logging.error(f"Model parallel setup failed: {e}")
        return model

def create_celery_app(config: DeploymentConfig) -> Celery:
    """Create Celery app for async task processing."""
    redis_url = f"redis://{config.redis_host}:{config.redis_port}/{config.redis_db}"
    
    celery_app = Celery(
        "quantum_consciousness",
        broker=redis_url,
        backend=redis_url
    )
    
    celery_app.conf.update(
        worker_prefetch_multiplier=1,
        worker_concurrency=config.workers,
        task_acks_late=True,
        task_reject_on_worker_lost=True
    )
    
    return celery_app

def configure_uvicorn(app: FastAPI, config: DeploymentConfig) -> None:
    """Configure Uvicorn server with production settings."""
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level,
        reload=config.reload,
        loop="uvloop",
        http="httptools",
        proxy_headers=True,
        forwarded_allow_ips="*"
    ) 