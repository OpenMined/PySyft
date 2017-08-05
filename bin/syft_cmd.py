import syft
import pickle
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from clint.arguments import Args
from clint.textui import puts, colored, indent

args = Args()

with indent(4, quote='>>>'):
    # puts(colored.red('Aruments passed in: ') + str(args.all))
    # puts(colored.red('Flags detected: ') + str(args.flags))
    # puts(colored.red('Files detected: ') + str(args.files))
    # puts(colored.red('NOT Files detected: ') + str(args.not_files))
    # puts(colored.red('Grouped Arguments: ') + str(dict(args.grouped)))

    command = str(args.all[0])

    if(command == 'generate_gradient'):

        model_path = dict(args.grouped)['-model'][0]
        data_path = dict(args.grouped)['-data'][0]
        gradient_path = dict(args.grouped)['-gradient'][0]

        puts(colored.blue('Command: ') + command)
        puts(colored.blue('Model Path: ') + model_path)
        puts(colored.blue('Data Path: ') + model_path)
        puts(colored.blue('Gradient Path: ') + gradient_path)

        from syft.he.Paillier import KeyPair
        from syft.nn.linear import LinearClassifier

    elif(command == 'create_model'):

        model_name = str(dict(args.grouped)['-name'][0])
        inputs = int(dict(args.grouped)['-inputs'][0])
        outputs = int(dict(args.grouped)['-outputs'][0])

        model_path = dict(args.grouped)['-path'][0]

        diabetes_classifier = syft.nn.linear.LinearClassifier(desc=model_name,n_inputs=inputs,n_labels=outputs)

        f = open(model_path+model_name.replace(" ","_")+".pickle",'wb')
        pickle.dump(diabetes_classifier,f)
        f.close()

    elif(command == 'generate_keypair'):

        local_args = set(dict(args.grouped).keys())

        if('-encryption' in local_args):
            encryption = str(dict(args.grouped)['-encryption'][0])
        else:
            encryption = 'paillier' #default

        if('-keylen' in local_args):
            keylen = int(dict(args.grouped)['-keylen'][0]) # 1024
        else:
            keylen = 1024 #default

        folder_path = str(dict(args.grouped)['-path'][0])

        if(encryption == 'paillier'):
            from syft.he.Paillier import KeyPair
            puts(colored.blue('Generating Keys...'))
            pubkey,prikey = KeyPair().generate(n_length=int(keylen))
            puts('DONE!')


            f = open(folder_path+"pubkey.pickle",'wb')
            pickle.dump(pubkey,f)
            f.close()

            f = open(folder_path+"prikey.pickle",'wb')
            pickle.dump(prikey,f)
            f.close()

        else:
            puts(colored.red('ENCRYPION ALGORITHM NOT FOUND'))



    else:
        usage = """
    Syft v0.1.0 - a library for Homomorphically Encrypted Deep Learning

    Usage: syft.py <command> [options]

    Commands:
        create_model        Initializes a new Linear model and saves it to disk
        generate_keypair    Generates a public and private Homomorphic Encryption Keys
        generate_gradient   Trains an existing model on some data, producing a diff (gradient)

    See more at: http://github.com/OpenMined/docs
        """
        puts(colored.red('COMMAND NOT FOUND'))
        puts(usage)


print
