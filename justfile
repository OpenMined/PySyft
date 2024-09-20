set dotenv-load

import 'justfiles/variables.just'
import 'justfiles/k3d.just'
import 'justfiles/devspace.just'
import 'justfiles/signoz.just'
import 'justfiles/utils.just'
import 'justfiles/cloud.just'
import 'justfiles/cluster.just'

@default:
    just --list
