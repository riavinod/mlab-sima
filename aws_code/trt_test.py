from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_linked_tensorrt_version
from tensorflow.compiler.tf2tensorrt.wrap_py_utils import get_loaded_tensorrt_version
   
compiled_version = get_linked_tensorrt_version()
loaded_version = get_loaded_tensorrt_version()
  
print("Linked TensorRT version: %s" % str(compiled_version))
print("Loaded TensorRT version: %s" % str(loaded_version))
                                                              
