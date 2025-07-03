from functools import wraps
import time
import logging
import asyncio
import inspect

# Create a global logger instance
app_logger = logging.getLogger('app')
app_logger.setLevel(logging.INFO)

if not app_logger.handlers:
    file_handler = logging.FileHandler('app.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)

def logger(original_function):
    global app_logger
    
    @wraps(original_function)
    def sync_wrapper(*args, **kwargs):
        # Get class context if available
        if '.' in original_function.__qualname__:
            class_name = original_function.__qualname__.split('.')[0]
            method_name = original_function.__qualname__.split('.')[1]
            context = f"{class_name}.{method_name}"
        else:
            context = original_function.__name__
        
        app_logger.info(f"[{context}] Calling with args: {args}, kwargs: {kwargs}")
        result = original_function(*args, **kwargs)
        app_logger.info(f"[{context}] Result: {result}")
        return result
    
    @wraps(original_function)
    async def async_wrapper(*args, **kwargs):
        # Get class context if available
        if '.' in original_function.__qualname__:
            class_name = original_function.__qualname__.split('.')[0]
            method_name = original_function.__qualname__.split('.')[1]
            context = f"{class_name}.{method_name}"
        else:
            context = original_function.__name__
        
        app_logger.info(f"[{context}] Calling with args: {args}, kwargs: {kwargs}")
        result = await original_function(*args, **kwargs)
        app_logger.info(f"[{context}] Result: {result}")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(original_function) else sync_wrapper


def timer(original_function):
    @wraps(original_function)
    def sync_wrapper(*args, **kwargs):
        # Get class context if available
        if '.' in original_function.__qualname__:
            class_name = original_function.__qualname__.split('.')[0]
            method_name = original_function.__qualname__.split('.')[1]
            context = f"{class_name}.{method_name}"
        else:
            context = original_function.__name__
        
        start_time = time.time()
        result = original_function(*args, **kwargs)
        end_time = time.time()
        app_logger.info(f"[{context}] took {end_time - start_time} seconds to run")
        return result
    
    @wraps(original_function)
    async def async_wrapper(*args, **kwargs):
        # Get class context if available
        if '.' in original_function.__qualname__:
            class_name = original_function.__qualname__.split('.')[0]
            method_name = original_function.__qualname__.split('.')[1]
            context = f"{class_name}.{method_name}"
        else:
            context = original_function.__name__
        
        start_time = time.time()
        result = await original_function(*args, **kwargs)
        end_time = time.time()
        app_logger.info(f"[{context}] took {end_time - start_time} seconds to run")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(original_function) else sync_wrapper
    """
    Convenience decorator to add both logging and timing to all methods in a class.
    
    Usage:
        @logged_and_timed_class
        class MyService:
            def method1(self):
                pass
    """
    return decorate_all_methods(logger, timer, exclude=exclude)