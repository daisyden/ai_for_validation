import sys
import inspect
import dis

def trace_calls(frame, event, arg):
    """Custom trace function that intercepts all function calls"""
    if event == 'call':
        # Get information about the called function
        code = frame.f_code
        func_name = code.co_name
        filename = code.co_filename
        line_no = frame.f_lineno
        
        # Get source code if available
        try:
            source_lines, start_line = inspect.getsourcelines(frame)
            current_line = line_no - start_line
            if 0 <= current_line < len(source_lines):
                source_code = source_lines[current_line].strip()
            else:
                source_code = "N/A"
        except:
            source_code = "N/A"
        
        print(f"📞 CALL: {func_name}() at {filename}:{line_no}")
        print(f"   Source: {source_code}")
        print(f"   Locals: {list(frame.f_locals.keys())}")
    
    elif event == 'line':
        # Called for each line execution
        print(f"📝 LINE: {frame.f_lineno} in {frame.f_code.co_name}")
    
    elif event == 'return':
        print(f"↩️  RETURN: {frame.f_code.co_name}() -> {arg}")
    
    elif event == 'exception':
        print(f"🚨 EXCEPTION: {arg[0].__name__}: {arg[1]}")
    
    return trace_calls  # Return itself to continue tracing

#############
#  Example  #
#############

# Enable tracing
sys.settrace(trace_calls)

# Example functions to trace
def calculate_sum(a, b):
    result = a + b
    return result

def process_data(data):
    total = 0
    for item in data:
        total += calculate_sum(item, 1)
    return total

# This will be traced
data = [1, 2, 3, 4, 5]
result = process_data(data)
print(f"Final result: {result}")

# Disable tracing
sys.settrace(None)
