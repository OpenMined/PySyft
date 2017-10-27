from distutils.core import setup, Extension


example_module = Extension('_PyPari',
                           sources=['PyPari_wrap.cxx'],
                           swig_opts=['-c++'],
                         include_dirs = ['/usr/local/include/'],
                           library_dirs = ['/usr/local/lib/'],
                           libraries = ['pari'],
                        # extra_compile_args = ["-std=c++11", "-lpari", "-mmacosx-version-min=10.7", "-pthread", "-O3", "-march=native", "-Wall", "-funroll-loops", "-Wno-unused-result","-I/usr/local/opt/openblas/include/"],
                        extra_compile_args = ["-std=c++11", "-pthread", "-O3", "-march=native", "-Wall", "-funroll-loops", "-Wno-unused-result", "-lpari"],
                           )

setup (name = 'PyPari',
       version = '0.2',
       author      = "SWIG Docs",
       description = """Simple swig linguamind from docs""",
       ext_modules = [example_module],
       py_modules = ["BV"],
       )
