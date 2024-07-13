import syft as sy
import os

def main():
    # Get the absolute path of the current script
    current_script_path = os.path.abspath(__file__)

    # Compute the parent directory of the current script
    parent_directory = os.path.dirname(current_script_path)

    compute_price_module_path = os.path.join(parent_directory, "node_pricing_structure.py")
    compute_price_func_name = "compute_price"

    sy.requires(">=0.8.6,<0.8.7")

    node = sy.orchestra.launch(
        name="test-domain-1",
        port=8080,
        dev_mode=True,
        reset=True,
        payment_required=True,
        node_payment_handle='node-payment-handle',
        payment_api='https://domain.tld/api/payments/',
        compute_price_module_path=compute_price_module_path,
        compute_price_func_name=compute_price_func_name
    )

    print(node)

    node.shutdown()

# Protect the entry point of the program
# This ensures that the multiprocessing code is not run
# when the module is imported as a part of the import of another module.
if __name__ == '__main__':
    main()
