from syft.codes import MSG_FIELD, GRID_EVENTS
import psutil


def _monitor(message: dict, conn_handler):
    """ Update peer status sending to the grid network some infos about this peer. """
    response = {MSG_FIELD.TYPE: GRID_EVENTS.MONITOR_ANSWER}

    response[MSG_FIELD.NODES] = conn_handler.nodes

    response[MSG_FIELD.CPU] = psutil.cpu_percent()
    response[MSG_FIELD.MEM_USAGE] = psutil.virtual_memory().percent
    models = {model_id: model.json() for model_id, model in conn_handler.worker.models.items()}
    response[MSG_FIELD.MODELS] = models

    def std_tags(tag):
        STD_TAGS = [
            "#fss_eq_plan_1",
            "#fss_eq_plan_2",
            "#fss_comp_plan_1",
            "#fss_comp_plan_2",
            "#xor_add_1",
            "#xor_add_2",
        ]
        if tag in STD_TAGS:
            return False
        return True

    response[MSG_FIELD.DATASETS] = list(
        filter(std_tags, conn_handler.worker.object_store._tag_to_object_ids.keys())
    )
    return response


def _create_webrtc_scope(message: dict, conn_handler):
    """ Send a p2p webrtc connection request to be forwarded by the grid network. """
    dest = message[MSG_FIELD.FROM]
    conn_handler.start_offer(dest)


def _accept_offer(message: dict, conn_handler):
    """ Receive a webrtc connection request sended by a peer and forwarded by the grid network. """
    dest = message.get(MSG_FIELD.FROM, None)
    content = message.get(MSG_FIELD.PAYLOAD, None)
    conn_handler.process_offer(dest, content)


def _process_webrtc_answer(message: dict, conn_handler):
    """ Process the peer answer. """
    dest = message.get(MSG_FIELD.FROM, None)
    content = message.get(MSG_FIELD.PAYLOAD, None)
    conn_handler.process_answer(dest, content)
