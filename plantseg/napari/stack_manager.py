from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GenericStack:
    name: str
    pipe: list = field(default_factory=list)
    original_voxel_size: list = field(default_factory=lambda: [1., 1., 1.])
    current_voxel_size: list = field(default_factory=lambda: [1., 1., 1.])


class ImageStack(GenericStack):
    data_range: Tuple[float, float] = (0., 1.)
    data_type: str = 'float32'


class LabelStack:
    data_type: str = 'uint16'


class StackState:

    def __int__(self):
        self._active_stacks = {}
        self.stack_types = {'ImageStack': ImageStack,
                            'LabelStack': LabelStack}

    def list_active_stacks(self):
        return list(self._active_stacks.keys())

    def set_stack_state(self, stack_name, stack_type: str = 'ImageStack'):
        if stack_name not in self._active_stacks:
            self._active_stacks[stack_name] = self.stack_types[stack_type](name=stack_name)

    def rm_stack_state(self, stack_name):
        if stack_name not in self._active_stacks:
            del self._active_stacks[stack_name]


stack_state = StackState()
