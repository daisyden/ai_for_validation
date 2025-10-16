import sys
import inspect
import pytest
import linecache


# Redefine trace_calls to also print the actual source line executed
def trace_calls(frame, event, arg):
    code_obj = frame.f_code
    func_name = code_obj.co_name
    filename = code_obj.co_filename
    lineno = frame.f_lineno

    if event == "call":
        src = linecache.getline(filename, lineno).strip()
        filename = filename.split("site-packages/")[-1]
        print(f"CALL: {func_name}() at {filename}:{lineno} -> {str}")
        print(f"\tLocals: {list(frame.f_locals.keys())}")
    elif event == "line":
        src = linecache.getline(filename, lineno).rstrip()
        filename = filename.split("site-packages/")[-1]
        print(f"LINE: {filename}:{lineno} in {func_name} -> {src}")
    # elif event == "return":
    #     print(f"RETURN: {func_name}() -> {arg}")
    #     print(f"RETURN: {func_name}()")
    elif event == "exception":
        print(f"EXCEPTION in {func_name}: {arg[0].__name__}: {arg[1]}")
    return trace_calls


################################
# Pytest hook 
################################

# To have pytest invoke pytest_runtest_protocol you do NOT call it directly.
# Make sure this module is loaded as a pytest plugin, then just run pytest.
#
# Option A (recommended): In some conftest.py (e.g. project root) add:
#     pytest_plugins = ["ai_for_validation.triage_bot_langgraph.common"]
#
# Option B: Load explicitly via command line:
#     pytest -p ai_for_validation.triage_bot_langgraph.common -s
#
# Option C: Programmatic (from another script):
#     import pytest
#     pytest.main(["-p", "ai_for_validation.triage_bot_langgraph.common", "-s", "tests/"])
#
# Use -s so the print from the hook is shown.
#
# Example:
#     pytest -p ai_for_validation.triage_bot_langgraph.tools.call_tracer -s tests/test_example.py::test_something
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    print(f"[triage] START {item.nodeid}")
    sys.settrace(trace_calls)
    try:
        yield
    finally:
        sys.settrace(None)
        print(f"[triage] END {item.nodeid}")
   


# # Enable tracing
# sys.settrace(trace_calls)

# # Example functions to trace
# def calculate_sum(a, b):
#     result = a + b
#     return result

# def process_data(data):
#     total = 0
#     for item in data:
#         total += calculate_sum(item, 1)
#     return total

# # This will be traced
# data = [1, 2, 3, 4, 5]
# result = process_data(data)
# print(f"Final result: {result}")

# # Disable tracing
# sys.settrace(None)


# def trace_calls(frame, event, arg):
#     """Custom trace function that intercepts all function calls"""
#     if event == 'call':
#         # Get information about the called function
#         code = frame.f_code
#         func_name = code.co_name
#         filename = code.co_filename
#         line_no = frame.f_lineno
        
#         # Get source code if available
#         try:
#             source_lines, start_line = inspect.getsourcelines(frame)
#             current_line = line_no - start_line
#             if 0 <= current_line < len(source_lines):
#                 source_code = source_lines[current_line].strip()
#             else:
#                 source_code = "N/A"
#         except:
#             source_code = "N/A"
        
#         print(f"CALL: {func_name}() at {filename}:{line_no}")
#         print(f"\tSource: {source_code}")
#         print(f"\tLocals: {list(frame.f_locals.keys())}")
    
#     elif event == 'line':
#         # Called for each line execution
#         print(f"LINE: {frame.f_lineno} in {frame.f_code.co_name}")
    
#     elif event == 'return':
#         print(f"RETURN: {frame.f_code.co_name}() -> {arg}")
    
#     elif event == 'exception':
#         print(f"EXCEPTION: {arg[0].__name__}: {arg[1]}")
    
#     return trace_calls  # Return itself to continue tracing
