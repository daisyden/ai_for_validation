import ast
import argparse

parser = argparse.ArgumentParser(
            prog='visit_ast',
            description='visit AST to add debug information in code',
            epilog='visit_ast <testfile> <testcase> <lineno>')

parser.add_argument('testfile')          # positional argument
parser.add_argument('testcase')          # positional argument
parser.add_argument('lineno')            # positional argument


args = parser.parse_args()

class VisitAST(ast.NodeTransformer):
    def __init__(self, line_func_dict):
        self.import_added = False
        self.line_func_dict = line_func_dict
        self.scope_stack = []
        self.visited = set()
        self.tensor_print = False
        

    def visit_Module(self, node):
        if not self.import_added:
            node.body = [
                ast.Import(names=[ast.alias(name='torch')]),
                ast.Import(names=[ast.alias(name='numpy', asname='np')])
            ] + node.body
            self.import_added = True
        node.body = [self.visit(n) for n in node.body]
        return node

    def visit_FunctionDef(self, node):
        if id(node) not in self.visited:
            for line in self.line_func_dict.keys():
                func = self.line_func_dict[line]
               
                if node.name == func and ( node.lineno <= line < max(getattr(n, 'lineno', node.lineno) for n in ast.walk(node)) ): 
                    self.scope_stack.append(node.name)
                    debug_print = self._create_print(
                            f"FUNC {node.name}({', '.join(arg.arg for arg in node.args.args)})"
                    )

                    if debug_print != None:
                        self.visited.add(id(node))

                    node.body = [debug_print] + [self.visit(n) for n in node.body]

                    self.scope_stack.pop()
                    break  

        return node

    def visit_Expr(self, node):
        return self._process_statement(node)

    def visit_Assign(self, node):
        return self._process_statement(node)

    def _process_statement(self, node):
        debug_nodes = []
        if id(node) not in self.visited:
            if len(self.scope_stack) != 0 and self.scope_stack[-1] in self.line_func_dict.values():
                if hasattr(node, 'lineno'):
                    stmt = ast.unparse(node).strip().replace('\n', ' ')[:80]
                    debug_nodes.append(self._create_print(f"LINE {node.lineno}: {stmt}"))

                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    debug_nodes.extend(self._debug_call(node.value))      
                elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    debug_nodes.extend(self._debug_call(node.value))

                for name in self._get_names(node):
                    debug_nodes.append(self._create_var_debug(name))

                if len(debug_nodes) != 0:
                    self.visited.add(id(node))

        return debug_nodes + [node]

    def _debug_call(self, call_node):
        debug_nodes = []
        func_name = ast.unparse(call_node.func)
        debug_nodes.append(self._create_print(f"\tCALL {func_name}"))
        debug_nodes.append(self._create_funcdef_inspect(f"{func_name}"))

        for i, arg in enumerate(call_node.args):
            if isinstance(arg, ast.Name):
                debug_nodes.append(self._create_var_debug(arg.id, f"ARG {i+1}"))

        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Name):
                debug_nodes.append(self._create_var_debug(kw.value.id, f"KWARG {kw.arg}"))

        if "assertEqual" in func_name and len(call_node.args) >= 2:
            if isinstance(call_node.args[0], ast.Name) and isinstance(call_node.args[0], ast.Name):
                debug_nodes.append(self._create_equal_checking(call_node.args[0], call_node.args[1]))

        return debug_nodes

    def _get_names(self, node):
        names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
        return names

    def _create_funcdef_inspect(self, func_name):
        """Safe object method inspect to get definition file."""
        debug_code = f"""
try:
    if len("{func_name}".split('.')) >= 2:
        import inspect
        _func_def_path = inspect.getsourcefile({func_name})
        _line_number = inspect.getsourcelines({func_name})[1]
        print(f"\tFunction {func_name} is defined at {{_func_def_path}}:{{_line_number}}", flush=True)
except Exception as e:
    print(f"\tCannot inpect {func_name} source file", flush=True)
"""
        return ast.parse(debug_code).body[0]

    def _create_var_debug(self, var_name, prefix=None):
        """Safe variable debugging that handles all types"""
        # If print tensor
        # print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: {{type({var_name}).__name__}} = {{repr({var_name})[:250]}} 
        # shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}}")
        debug_code = f"""
try:
    if isinstance({var_name}, (torch.Tensor,)):
        print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}} device={{str({var_name}.device)}}", flush=True)
    elif isinstance({var_name}, (np.ndarray, )):
        print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}}", flush=True)
    elif hasattr({var_name}, 'shape') and hasattr({var_name}, 'dtype'):
        print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}}", flush=True)
    else:
        print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: {{type({var_name}).__name__}} = {{repr({var_name})[:250]}} ", flush=True)
except Exception as e:
    print(f"\t{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: [Error during inspection: {{str(e)}}]", flush=True)
"""
        return ast.parse(debug_code).body[0]
    
    def _create_print(self, message):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id='print', ctx=ast.Load()),
            args=[ast.Constant(value=message)],
            keywords=[ast.keyword(arg='flush', value=ast.Constant(value=True))]
        ))
    
    def _create_equal_checking(self, arg1, arg2):
        debug_code =f"""
try:
    if isinstance({arg1.id}, (torch.Tensor, np.ndarray)) and isinstance({arg2.id}, (torch.Tensor, np.ndarray)):
        _diff_indices = ({arg1.id} != {arg2.id}).nonzero()
        print(f"\t Diff_indices: {{_diff_indices}}", flush=True)
        # for _indices in _diff_indices:
        #     a = {arg1.id}[_indices]
        #     b = {arg2.id}[_indices]
        #     print(f"\t\t {{_indices}}: {{a}} .vs {{b}}")        
except Exception as e:
    print(f"\tGet differing indices failed", flush=True)
"""
        return ast.parse(debug_code).body[0]

def debug_transform(source_code, function_dict):
    import copy
    _function_dict = copy.deepcopy(function_dict)

    tree = ast.parse(source_code)
    func_def = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_def[node.name] = node.lineno

    for index, (_line, _func) in enumerate(function_dict.items()):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == _func:
                for node in ast.walk(node):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            #print(f"#### {node.func.id} is called by {_func}\n")
                            if node.func.id in func_def.keys():                                
                                _function_dict.update({node.func.lineno: node.func.id,})
                                #print(_function_dict)
                        if isinstance(node.func, ast.Attribute):
                            if isinstance(node.func.value, ast.Name) and node.func.attr in func_def.keys():
                                #print(f"#### {_function_dict} {node.func.value.id}.{node.func.attr} is called by {_func}\n")                          
                                _function_dict.update({func_def[node.func.attr]: node.func.attr,})
                                #print(_function_dict)
    transformer = VisitAST(_function_dict)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)

def main():
    try:
        with open(args.testfile, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("Error: The file 'my_file.txt' was not found.")
    
    source = content
    
    func = {int(args.lineno): args.testcase}
    print(debug_transform(source, func))


if __name__ == "__main__":
    main()
