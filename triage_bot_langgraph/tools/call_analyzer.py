import ast
import inspect
import sys
import os
import json
import importlib.util
from typing import Dict, List, Any, Optional
import textwrap

class DynamicFunctionAnalyzer:
    def __init__(self):
        self.imports: List[Dict[str, Any]] = []
        self.decorators: List[Dict[str, Any]] = []
        self.subfunctions: List[Dict[str, Any]] = []
        self.method_calls: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.defined_functions: Dict[str, Dict[str, Any]] = {}
        self.imported_names: Dict[str, str] = {}
        self.current_scope: List[str] = []
        self.current_file: str = ""
        self.module_cache: Dict[str, ast.Module] = {}
        self.spec_cache: Dict[str, importlib.util.ModuleSpec] = {}

    def analyze_function(self, func) -> Dict[str, Any]:
        """Analyze a function dynamically with definition points"""
        try:
            source_lines, start_line = inspect.getsourcelines(func)
            source_code = textwrap.dedent(''.join(source_lines))
            if source_code.lstrip().startswith('@'):
                # Get additional lines until we find the function definition
                lines = inspect.getblock(source_lines)
                source_code = textwrap.dedent(''.join(lines))
            file_path = os.path.abspath(inspect.getsourcefile(inspect.unwrap(func)))
            line_no = inspect.unwrap(func).__code__.co_firstlineno if hasattr(inspect.unwrap(func), '__code__') else 0
            source = source_code
        except (TypeError, OSError) as e:
            return {"error": f"Could not get source: {str(e)}"}
        
        self.current_file = os.path.abspath(file_path) if file_path else ""
        tree = ast.parse(source)
        
        import pdb
        pdb.set_trace()
        # Collect imports from the module first
        module = inspect.getmodule(func)
        if module and hasattr(module, '__file__'):
            self._collect_module_imports(module.__file__)
        
        # Analyze the function
        self._analyze_function_node(tree, file_path, line_no, func)

        return {
            "function_info": self._get_function_info(func),
            #"imports": self.imports,
            "decorators": self.decorators,
            "subfunctions": self.subfunctions,
            "method_calls": self.method_calls,
            "function_calls": self.function_calls
        }

    def _get_module_definition(self, module_name: str, module_path: str) -> Optional[Dict[str, Any]]:
        """Get the module-level definition information"""
        if not module_path:
            return None

        try:
            # Try to get the module docstring and basic info
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return {
                    "file": module_path,
                    "type": "module",
                    "name": module_name,
                    "doc": inspect.getdoc(module),
                    "line": 1  # Modules start at line 1
                }
        except Exception:
            pass

        # Fallback to basic file info
        return {
            "file": module_path,
            "type": "module",
            "name": module_name,
            "line": 1
        }

    def _import_module_from_path(self, module_path: str) -> Optional[Any]:
        """Import a module from a specific file path"""
        try:
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            if module_path in self.spec_cache:
                spec = self.spec_cache[module_path]
            else:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                self.spec_cache[module_path] = spec
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            print(f"Failed to import module from {module_path}: {str(e)}")
        return None

    def _get_function_info(self, func) -> Dict[str, Any]:
        """Get detailed information about the target function"""
        return {
            "name": func.__name__,
            "file": inspect.getsourcefile(func),
            "line": func.__code__.co_firstlineno,
            "is_async": inspect.iscoroutinefunction(func),
            "decorators": self._get_function_decorators(func)
        }

    def _collect_module_imports(self, module_path: str) -> None:
        """Collect imports from the module with definition points"""
        if module_path not in self.module_cache:
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    module_source = f.read()
                self.module_cache[module_path] = ast.parse(module_source)
            except Exception:
                return

        for node in ast.walk(self.module_cache[module_path]):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._process_import(node, alias.name, alias.asname, module_path)
            elif isinstance(node, ast.ImportFrom):
                self._process_import_from(node, module_path)

    def _process_import(self, node: ast.AST, name: str, alias: Optional[str], source_path: str) -> None:
        """Process a regular import statement"""
        imported_name = alias or name
        self.imported_names[imported_name] = name
        
        # Try to find the actual module
        module_path = self._find_module_path(name, source_path)
        definition = self._get_module_definition(name, module_path) if module_path else None
        
        self.imports.append({
            "type": "import",
            "name": name,
            "alias": alias,
            "line": node.lineno,
            "source_file": source_path,
            "definition": definition
        })

    def _process_import_from(self, node: ast.ImportFrom, source_path: str) -> None:
        """Process a from-import statement"""
        module = node.module or ""
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            imported_name = alias.asname or alias.name
            self.imported_names[imported_name] = full_name
            
            # Try to find the actual definition
            definition = self._find_imported_definition(module, alias.name, node.level, source_path)
            
            self.imports.append({
                "type": "from_import",
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "level": node.level,
                "line": node.lineno,
                "source_file": source_path,
                "definition": definition
            })

    def _find_imported_definition(self, module: str, name: str, level: int, source_path: str) -> Optional[Dict[str, Any]]:
        """Find the definition point of an imported name with import_from_path"""
        try:
            if level > 0:
                base_path = os.path.dirname(source_path)
                for _ in range(level - 1):
                    base_path = os.path.dirname(base_path)
                module_path = os.path.join(base_path, module.replace('.', os.sep) + '.py')
            else:
                module_path = self._find_module_path(module, source_path)
            
            if not module_path or not os.path.exists(module_path):
                return None

            # Try to import the module to get accurate symbol resolution
            imported_module = self._import_module_from_path(module_path)
            if imported_module:
                target = getattr(imported_module, name, None)
                if target:
                    return self._get_definition_from_object(target, module_path)

            # Fallback to AST analysis
            if module_path not in self.module_cache:
                with open(module_path, 'r', encoding='utf-8') as f:
                    self.module_cache[module_path] = ast.parse(f.read())

            # Find the definition in the imported module
            for node in ast.walk(self.module_cache[module_path]):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    return {
                        "file": module_path,
                        "line": node.lineno,
                        "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        #"code": ast.get_source_segment(self.module_cache[module_path], node)
                    }
        except Exception as e:
            print(f"Error finding imported definition: {str(e)}")
        return None

    def _get_definition_from_object(self, obj, module_path: str) -> Optional[Dict[str, Any]]:
        """Get definition info from a live Python object"""
        try:
            file_path = inspect.getsourcefile(obj)
            lines, line_no = inspect.getsourcelines(obj)
            return {
                "file": file_path or module_path,
                "line": line_no,
                "type": "class" if inspect.isclass(obj) else "function",
                "code": "".join(lines)
            }
        except Exception:
            return None

    def _find_module_path(self, module_name: str, source_path: str) -> Optional[str]:
        """Find the file path for a module, considering relative imports"""
        try:
            # First try using importlib
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return spec.origin
            
            # Fallback to filesystem search relative to source
            base_dir = os.path.dirname(source_path)
            module_path = os.path.join(base_dir, module_name.replace('.', os.sep) + '.py')
            if os.path.exists(module_path):
                return module_path
            
            package_path = os.path.join(base_dir, module_name.replace('.', os.sep), '__init__.py')
            if os.path.exists(package_path):
                return package_path
            
            # Search in Python path
            for path in sys.path:
                if not path.strip():
                    continue
                
                module_path = os.path.join(path, module_name.replace('.', os.sep) + '.py')
                if os.path.exists(module_path):
                    return module_path
                
                package_path = os.path.join(path, module_name.replace('.', os.sep), '__init__.py')
                if os.path.exists(package_path):
                    return package_path
        except Exception:
            pass
        return None

    def _get_function_decorators(self, func) -> List[Dict[str, Any]]:
        """Get decorators with their definition points"""
        decorators = []
        try:
            # Get decorators from AST
            source = inspect.getsource(func)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func.__name__:
                    for decorator in node.decorator_list:
                        decorator_name = ast.unparse(decorator)
                        definition = self._find_decorator_definition(decorator)
                        decorators.append({
                            "name": decorator_name,
                            "line": decorator.lineno,
                            "file": inspect.getsourcefile(func),
                            "definition": definition
                        })
                    break
        except Exception:
            pass
        return decorators

    def _find_decorator_definition(self, decorator_node: ast.AST) -> Optional[Dict[str, Any]]:
        """Find where a decorator is defined"""
        if isinstance(decorator_node, ast.Name):
            # Check if it's an imported name
            if decorator_node.id in self.imported_names:
                imported_name = self.imported_names[decorator_node.id]
                return self._find_imported_definition(
                    imported_name.rsplit('.', 1)[0] if '.' in imported_name else '',
                    imported_name.rsplit('.', 1)[-1],
                    0,
                    self.current_file
                )
            
            # Check local definitions
            for file_path, tree in self.module_cache.items():
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == decorator_node.id:
                        return {
                            "file": file_path,
                            "line": node.lineno,
                            "type": "class" if isinstance(node, ast.ClassDef) else "function",
                            #"code": ast.get_source_segment(tree, node)
                        }
        return None

    def _analyze_function_node(self, tree: ast.AST, file_path: str, line_offset: int, func) -> None:
        """Analyze the function's AST with definition points"""
        self.current_scope = [func.__name__]
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func.__name__:
                    # Process decorators for the main function
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            self._process_call(decorator, file_path)
                        else:
                            self._process_decorator(decorator, file_path)
                else:
                    # Process subfunctions
                    self._process_subfunction(node, file_path)
            
            elif isinstance(node, ast.Call):
                self._process_call(node, file_path)

    def _process_decorator(self, decorator_node: ast.AST, file_path: str) -> None:
        """Process a decorator with definition point"""
        decorator_name = ast.unparse(decorator_node)
        definition = self._find_decorator_definition(decorator_node)
        
        self.decorators.append({
            "name": decorator_name,
            "line": decorator_node.lineno,
            "file": file_path,
            "definition": definition
        })

    def _process_subfunction(self, node: ast.AST, file_path: str) -> None:
        """Process nested functions with definition points"""
        subfunction_info = {
            "name": node.name,
            "type": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
            "line": node.lineno,
            "file": file_path,
            "scope": " > ".join(self.current_scope),
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "definition": {
                "file": file_path,
                "line": node.lineno,
                #"code": ast.get_source_segment(self.module_cache.get(file_path, None), node)
            }
        }
        self.subfunctions.append(subfunction_info)

    def _process_call(self, node: ast.Call, file_path: str) -> None:
        """Process function/method calls with definition points"""
        call_info = {
            "line": node.lineno,
            "file": file_path,
            "scope": " > ".join(self.current_scope),
            "args_count": len(node.args) + len(node.keywords)
        }

        if isinstance(node.func, ast.Attribute):
            # Method call - try to find the class definition
            method_name = node.func.attr
            obj_expr = ast.unparse(node.func.value)
            # import pdb
            # pdb.set_trace()
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                definition = self._find_method_definition(obj_name, method_name)
            else:
                definition = None
            
            call_info.update({
                "type": "method_call",
                "object": obj_expr,
                "method": method_name,
                "full_name": f"{obj_expr}.{method_name}",
                "definition": definition
            })
            self.method_calls.append(call_info)

        elif isinstance(node.func, ast.Name):
            # Regular function call
            func_name = node.func.id
            definition = self._find_function_definition(func_name)
            
            call_info.update({
                "type": "function_call",
                "name": func_name,
                "definition": definition
            })
            self.function_calls.append(call_info)

    def _find_method_definition(self, obj_name: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Find where a method is defined"""
        # Check if it's an imported class
        if obj_name in self.imported_names:
            imported_name = self.imported_names[obj_name]
            parts = imported_name.split('.')
            module_path = self._find_module_path('.'.join(parts[:-1]) if len(parts) > 1 else imported_name, self.current_file)
            
            if module_path:
                return self._find_class_method_definition(module_path, parts[-1], method_name)
        
        # Check local definitions
        for file_path, tree in self.module_cache.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == obj_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                            return {
                                "file": file_path,
                                "line": item.lineno,
                                "type": "method",
                                "class": obj_name,
                                #"code": ast.get_source_segment(tree, item)
                            }
        return None

    def _find_function_definition(self, func_name: str) -> Optional[Dict[str, Any]]:
        """Find where a function is defined"""
        # Check if it's an imported name
        if func_name in self.imported_names:
            imported_name = self.imported_names[func_name]
            if '.' in imported_name:
                module, name = imported_name.rsplit('.', 1)
                module_path = self._find_module_path(module, self.current_file)
                if module_path:
                    return self._find_definition_in_module(module_path, name)
            else:
                module_path = self._find_module_path(imported_name, self.current_file)
                if module_path:
                    return self._find_definition_in_module(module_path, imported_name)
        
        # Check local definitions
        for file_path, tree in self.module_cache.items():
            definition = self._find_definition_in_module(file_path, func_name, tree)
            if definition:
                return definition
        return None

    def _find_definition_in_module(self, file_path: str, name: str, tree: Optional[ast.Module] = None) -> Optional[Dict[str, Any]]:
        """Find a definition in a specific module"""
        if tree is None:
            if file_path not in self.module_cache:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    self.module_cache[file_path] = tree
                except Exception:
                    return None
            else:
                tree = self.module_cache[file_path]
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return {
                    "file": file_path,
                    "line": node.lineno,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    #"code": ast.get_source_segment(tree, node)
                }
        return None

    def _find_class_method_definition(self, module_path: str, class_name: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Find a method definition in a class from another module"""
        if module_path not in self.module_cache:
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    self.module_cache[module_path] = ast.parse(f.read())
            except Exception:
                return None
        
        for node in ast.walk(self.module_cache[module_path]):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        return {
                            "file": module_path,
                            "line": item.lineno,
                            "type": "method",
                            "class": class_name,
                            #"code": ast.get_source_segment(self.module_cache[module_path], item)
                        }
        return None

def analyze_function_dynamically(func) -> Dict[str, Any]:
    """Convenience function to analyze a function and return results"""
    analyzer = DynamicFunctionAnalyzer()
    return analyzer.analyze_function(func)

# Example usage
if __name__ == "__main__":
#     # Create test files
#     os.makedirs("test_package", exist_ok=True)
    
#     # Create a module with a class
#     with open("test_package/module_with_class.py", 'w', encoding='utf-8') as f:
#         f.write("""
# class ExternalClass:
#     def external_method(self, x):
#         return x * 3
        
#     @staticmethod
#     def static_method(x):
#         return x * 4
# """)

#     # Create a test file that imports the class
#     with open("test_function.py", 'w', encoding='utf-8') as f:
#         f.write("""
# from test_package.module_with_class import ExternalClass

# def example_function():
#     obj = ExternalClass()
#     result = obj.external_method(5)
#     static_result = ExternalClass.static_method(6)
#     return result + static_result
# """)

#     # Analyze and print results
#     import test_function
#     analysis = analyze_function_dynamically(test_function.example_function)

    os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
    os.environ["PYTHONPATH"] = "/home/daisyden/upstream/test_ops_s1/test:/home/daisyden/miniforge3/envs/test_ops_for_lg/lib/python3.10/"
    path_name = '/home/daisyden/upstream/distributed_s2/test/test_ops.py'
    path, _ = os.path.splitext(path_name)
    file_name = path.split('/')[-1] 
    #import importlib.util
    mod = importlib.import_module(file_name)

    # Analyze the function dynamically
    analysis = analyze_function_dynamically(mod.TestCommonCPU.test_numpy_ref_allclose_cpu_complex128)
  
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    # Clean up
    # os.remove("test_function.py")
    # os.remove("test_package/module_with_class.py")
    # os.rmdir("test_package")