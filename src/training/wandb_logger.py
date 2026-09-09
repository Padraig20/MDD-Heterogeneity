import wandb
import os

ENTITY  = "your-entity"
PROJECT = "your-project"

if "WANDB_KEY" not in os.environ:
    raise EnvironmentError("You should *really* set the WANDB_KEY environment variable!!!")

wandb.login(key=os.getenv("WANDB_KEY"))
class WandBLogger:

    def __init__(
        self,
        enabled=True,
        run_name: str = None,
        log_every: int = 50,
    ) -> None:
        
        self.enabled = enabled
        self.log_every = log_every
        self._batch_step = 0

        if self.enabled:
            wandb.init(entity=ENTITY,
                       project=PROJECT)
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  
            
    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)

    def log_batch(self, log_dict: dict) -> None:
        """Log a per-batch metric only every ``log_every`` training steps."""
        if not self.enabled:
            return
        self._batch_step += 1
        if self.log_every <= 0:
            return
        if self._batch_step % self.log_every == 0:
            wandb.log(log_dict)
 
    def finish(self):
        if self.enabled:
            wandb.finish()
