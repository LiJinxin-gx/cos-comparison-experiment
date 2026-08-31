"""
Von Neumann Architecture Agent Engine.

Core insight: treat INSTRUCTIONS (code execution) as DATA stored in a
database. A FIXED code reader executes stored instructions at runtime.
This achieves:
  - Fixed code implements PLURAL logic (logic is data, not code)
  - Execution logic can be GENERATED and MODIFIED without changing code
  - Behavior is controlled by external config / prompt, not hardcoded

This mirrors the von Neumann architecture: program and data share the same
memory space. Here, "program" = behavior instructions stored as data,
"CPU" = the fixed interpreter that reads and executes them.

Instruction set (minimal, extensible):
  - STORE key value     : store data in memory
  - LOAD key            : load data from memory
  - MATCH tensor        : match against stored templates
  - SEARCH query        : search memory
  - GENERATE params     : generate output from templates
  - BRANCH condition    : conditional branch
  - JUMP addr           : unconditional jump
  - OUTPUT value        : emit output
"""
from core.cosine import cosine, as_tensor, flatten
from .memory import HierarchicalMemory


class VonNeumannAgent:
    """Fixed-code interpreter for data-stored behavior programs.

    The agent itself has NO hardcoded behavior — all behavior comes from
    instruction programs stored as data (lists of instruction tuples).
    """

    def __init__(self, memory_levels=3):
        self.memory = HierarchicalMemory(n_levels=memory_levels)
        self.registers = {}  # general-purpose registers
        self.pc = 0          # program counter
        self.output_buffer = []
        self.instruction_set = {
            "STORE": self._instr_store,
            "LOAD": self._instr_load,
            "MATCH": self._instr_match,
            "SEARCH": self._instr_search,
            "GENERATE": self._instr_generate,
            "BRANCH": self._instr_branch,
            "JUMP": self._instr_jump,
            "OUTPUT": self._instr_output,
            "COMPARE": self._instr_compare,
        }

    def load_program(self, instructions):
        """Load a behavior program (list of instruction tuples).

        Each instruction: (opcode, *args)
        Example: [("STORE", "goal", "surf_web"), ("SEARCH", "goal"), ...]
        """
        self.program = list(instructions)
        self.pc = 0
        self.output_buffer = []

    def run(self, max_steps=1000):
        """Execute the loaded program. Fixed code, variable behavior."""
        steps = 0
        while self.pc < len(self.program) and steps < max_steps:
            instr = self.program[self.pc]
            opcode = instr[0]
            args = instr[1:]
            if opcode in self.instruction_set:
                self.instruction_set[opcode](*args)
            else:
                self.output_buffer.append(f"UNKNOWN: {opcode}")
            self.pc += 1
            steps += 1
        return self.output_buffer

    def _instr_store(self, key, value):
        self.registers[key] = value

    def _instr_load(self, key):
        return self.registers.get(key, None)

    def _instr_match(self, tensor, level=0):
        results = self.memory.search(tensor, level=level, top_k=3)
        self.registers["last_match"] = results
        return results

    def _instr_search(self, query):
        if isinstance(query, str) and query in self.registers:
            query = self.registers[query]
        if isinstance(query, str):
            query = [ord(c) for c in query]
        elif isinstance(query, (list, tuple)):
            # Flatten list of strings/numbers into numeric tensor
            flat = []
            for item in query:
                if isinstance(item, str):
                    flat.extend(ord(c) for c in item)
                else:
                    flat.append(float(item))
            query = flat
        results = self.memory.search(query, top_k=5)
        self.registers["last_search"] = results
        return results

    def _instr_generate(self, template_key, params=None):
        """Generate output from a stored template.

        Simplified: retrieves template and applies params as modifiers.
        """
        template = self.registers.get(template_key, "")
        if params:
            for k, v in params.items():
                template = template.replace(f"{{{k}}}", str(v))
        self.output_buffer.append(template)
        return template

    def _instr_branch(self, condition, target):
        """Conditional branch: if condition is truthy, jump to target."""
        cond_value = self.registers.get(condition, condition)
        if cond_value:
            self.pc = target - 1  # -1 because loop increments after

    def _instr_jump(self, target):
        self.pc = target - 1

    def _instr_output(self, value):
        if isinstance(value, str) and value in self.registers:
            value = self.registers[value]
        self.output_buffer.append(value)

    def _instr_compare(self, key_a, key_b, target):
        """Compare two registers; if equal, branch to target."""
        a = self.registers.get(key_a)
        b = self.registers.get(key_b)
        if a == b:
            self.pc = target - 1

    def teach(self, examples):
        """Learn behavior from examples (few-shot learning).

        examples: list of (input_tensor, output_program) pairs.
        Stores input-output mappings in memory for later retrieval.
        """
        for inp, program in examples:
            self.memory.store(inp, label=str(program), level=0)

    def infer_behavior(self, input_tensor):
        """Infer behavior from memory: find similar example, use its program."""
        results = self.memory.search(input_tensor, level=0, top_k=1)
        if results:
            _, _, label, _ = results[0]
            # label is the program string; in practice would deserialize
            return label
        return None


def make_surf_program(interest_keywords, max_depth=5):
    """Factory: create a web-surfing behavior program from config.

    This demonstrates behavior-as-data: the same agent runs different
    programs depending on external config.
    """
    program = [
        ("STORE", "interests", interest_keywords),
        ("STORE", "depth", 0),
        ("STORE", "max_depth", max_depth),
        # Main loop
        ("SEARCH", "interests"),
        ("OUTPUT", "Exploring based on interests..."),
        ("LOAD", "depth"),
        ("COMPARE", "depth", "max_depth", 99),  # branch to end if depth == max
        ("STORE", "depth", "depth+1"),  # simplified increment
        ("JUMP", 4),  # loop back to search
        # End
        ("OUTPUT", "Surfing complete"),
    ]
    return program
