import syft as sy

def main(): 
  sy.requires(">=0.8.6,<0.8.7")

  node = sy.orchestra.launch(
      name="my-domain",
      port=8080,
      create_producer=True,
      n_consumers=1,
      dev_mode=True,
      reset=True, # resets database
  )

  print(node)

  node.shutdown()

# Protect the entry point of the program
# This ensures that the multiprocessing code is not run 
# when the module is imported as a part of the import of another module.
if __name__ == '__main__':
  main()
