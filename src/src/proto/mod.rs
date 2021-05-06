pub mod core {
    pub mod auth {
        tonic::include_proto!("syft.core.auth");
    }

    pub mod common {
        tonic::include_proto!("syft.core.common");
    }

    pub mod io {
        tonic::include_proto!("syft.core.io");
    }

    pub mod node {
        pub mod common {
            tonic::include_proto!("syft.core.node.common");
            pub mod action {
                tonic::include_proto!("syft.core.node.common.action");
            }

            pub mod remote_dataloader {
                tonic::include_proto!("syft.core.node.common.remote_dataloader");
            }

            pub mod service {
                tonic::include_proto!("syft.core.node.common.service");
            }
        }

        pub mod domain {
            pub mod service {
                // tonic::include_proto!("syft.core.node.domain.service");
            }
        }
    }

    pub mod plan {
        tonic::include_proto!("syft.core.plan");
    }

    pub mod pointer {
        tonic::include_proto!("syft.core.pointer");
    }

    pub mod store {
        tonic::include_proto!("syft.core.store");
    }
}

pub mod lib {
    pub mod numpy {
        tonic::include_proto!("syft.lib.numpy");
    }

    pub mod torch {
        tonic::include_proto!("syft.lib.torch");
    }

    pub mod python {
        pub mod collections {
            tonic::include_proto!("syft.lib.python.collections");
        }

        tonic::include_proto!("syft.lib.python");
    }
}

pub mod util {
    tonic::include_proto!("syft.util");
}
