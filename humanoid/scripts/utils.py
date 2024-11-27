import os

def dump_onnx_inputs(onnx_input, dump_input_folder_path, dump_input_start_index):
    for name, array in onnx_input.items():
        model_dump_each_input_subfolder_name = os.path.join(
            dump_input_folder_path, name
        )
        os.makedirs(
            model_dump_each_input_subfolder_name, exist_ok=True
        )
        input_bin_path = os.path.join(
            model_dump_each_input_subfolder_name,
            f"{name + str(dump_input_start_index)}.bin",
        )
        array.tofile(input_bin_path)
        