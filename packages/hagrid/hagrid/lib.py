import os

def some_function():
    print('Running lib.some_function')

def motorcycle():
    print("""
                                             `
                                         `.+yys/.`
                                       ``/NMMMNNs`
                                    `./shNMMMMMMNs``    `..`
                                  `-smNMMNNMMMMMMN/.``......`
                                `.yNMMMMNmmmmNNMMm/.`....`
                              `:sdNMMMMMMNNNNddddds-`.`` `--. `
                           `.+dNNNNMMMMMMMMMNNNNmddohmh//hddy/.```..`
                          `-hNMMMMMMMMMMMMNNdmNNMNNdNNd:sdyoo+/++:..`
                        ../mMMMMMMMMMMMMMMNNmmmmNMNmNNmdmd/hNNNd+:`
                        `:mMNNMMMMMMMMMMMMNMNNmmmNNNNNdNNd/NMMMMm::
                       `:mMNNNMMMMMMMMMMMMMMMNNNNdNMNNmmNd:smMMmh//
                     ``/mMMMMMMMMMMMMMMMMMMMMMMNmdmNNMMNNNy/osoo/-`
                    `-sNMMMMMMMMMMMMMMMMMMMMMMMMNNmmMMMMNh-....`
                   `/dNMMMMMMMMMMMMMMMMMMMMMMMMMMMNNMMMNy.`
                ``.omNNMMMMMMMMMMMMNMMMMMMMNmmmmNNMMMMN+`
                `:hmNNMMMMMMMMMMMNo/ohNNNNho+os+-+hNys/`
                -mNNNNNNMMMMMMMMm+``-yNdd+/mMMMms.-:`
                .+dmNNNNMMMMMMNd:``:dNN+y`oMMMMMm-.`
                `+dmmmNNNmmmmy+.   `-+m/s/+MMMMm/--
               `+mmmhNy/-...```     ``-.-sosyys+/-`
            ``.smmmsoo``               .oh+-:/:.
          `.:odmdh/````             `.+d+``````
     ```/sydNdhy+.`              ``-sds.
    `:hdmhs::-````               `oNs.`
```.sdmh/``                    `-ym+`
 ``ssy+`                     `-yms.`
   ``                      `:hNy-``
   `                     `-yMN/```
                       `-yNhy-
                     `/yNd/`
                   `:dNMs``
                 `.+mNy/.`
              `.+hNMMs``
             `:dMMMMh.`""")

    rows, columns = os.popen('stty size', 'r').read().split()

    if int(columns) >= 91:

        print("""
 _   _       _     _                 _   _       _     _     _   _                       _ 
| | | |     | |   | |               | | (_)     | |   | |   | | | |                     | |
| |_| | ___ | | __| |   ___  _ __   | |_ _  __ _| |__ | |_  | |_| | __ _ _ __ _ __ _   _| |
|  _  |/ _ \| |/ _` |  / _ \| '_ \  | __| |/ _` | '_ \| __| |  _  |/ _` | '__| '__| | | | |
| | | | (_) | | (_| | | (_) | | | | | |_| | (_| | | | | |_  | | | | (_| | |  | |  | |_| |_|
\_| |_/\___/|_|\__,_|  \___/|_| |_|  \__|_|\__, |_| |_|\__| \_| |_/\__,_|_|  |_|   \__, (_)
                                            __/ |                                   __/ |  
                                           |___/                                   |___/     
            """)
    else:
        print("""
 _   _       _     _                 _   _                       _ 
| | | |     | |   | |               | | | |                     | |
| |_| | ___ | | __| |   ___  _ __   | |_| | __ _ _ __ _ __ _   _| |
|  _  |/ _ \| |/ _` |  / _ \| '_ \  |  _  |/ _` | '__| '__| | | | |
| | | | (_) | | (_| | | (_) | | | | | | | | (_| | |  | |  | |_| |_|
\_| |_/\___/|_|\__,_|  \___/|_| |_| \_| |_/\__,_|_|  |_|   \__, (_)
                                                            __/ |  
                                                           |___/    
        """)