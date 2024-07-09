import syft as sy

def main():
  sy.requires(">=0.8.6,<0.8.7")

  node = sy.orchestra.launch(
      name="test-domain-1",
      port=8080,
      dev_mode=True,
      reset=True
  )

  print(node)

  node.shutdown()

# Protect the entry point of the program
# This ensures that the multiprocessing code is not run
# when the module is imported as a part of the import of another module.
if __name__ == '__main__':
  main()
