import importlib
import sys

import torch


def test_e1_import_chain_does_not_call_torch_compile(monkeypatch):
    def _forbidden_compile(*args, **kwargs):
        raise AssertionError("E1 不应在模块导入阶段调用 torch.compile")

    monkeypatch.setattr(torch, "compile", _forbidden_compile)

    for module_name in list(sys.modules):
        if module_name == "E1.modeling" or module_name.startswith("E1.model"):
            sys.modules.pop(module_name, None)

    importlib.import_module("E1.model.attention")
    importlib.import_module("E1.modeling")
