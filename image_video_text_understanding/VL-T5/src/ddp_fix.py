def _run_ddp_forward(self, *inputs, **kwargs):
    with self._inside_ddp_forward():
        return self.module.train_step(*inputs, **kwargs)  # type: ignore[index]


def ddp_forward(model, *inputs, **kwargs):
    model._run_ddp_forward = _run_ddp_forward.__get__(model)
    return model(*inputs, **kwargs)
