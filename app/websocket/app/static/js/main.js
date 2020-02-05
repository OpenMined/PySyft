
var getUrl = window.location;
var baseUrl = getUrl.protocol + "//" + getUrl.host + "/" + getUrl.pathname.split('/')[1];

// AXIOS API Functions:
axios.baseUrl = baseUrl

async function get_identity_from_server() {
  try {
    const response = await axios.get('/identity/');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

async function get_status_from_server() {
  try {
    const response = await axios.get('/status/');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}


async function get_models_from_server() {
  try {
    const response = await axios.get('/detailed_models_list/');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

async function get_dataset_tags() {
  try {
    const response = await axios.get('/dataset-tags');
    return Promise.resolve(response);
  } catch (error) {
    console.log(error);
    return Promise.resolve(error);
  }
}

async function get_workers_from_server() {
  try {
    const response = await axios.get('/workers/');
    return Promise.resolve(response)
  } catch (error) {
    console.error(error);
    return Promise.resolve(error)
  }
}

// Vue Variable Setters:

function set_online_status(status) {
  online_status.status = status
}

function set_name_of_node(name) {
  name_of_node.name = name
}

function set_models_in_table(models) {
  models_list.models = models
}

function set_workers_in_table(workers) {
  workers_list.workers = workers
}

function set_tags_in_table(data_tags) {
  tags_list.data_tags = data_tags
}

// VUE OBJECT BINDINGS:
var models_list = new Vue({
  el: '#m-for-models-list',
  delimiters: ['[[', ']]'],
  data: {
    models: []
  }
})

var tags_list = new Vue({
  el: '#v-for-tags-list',
  delimiters: ['[[', ']]'],
  data: {
    data_tags: ['oi']
  }
})

var workers_list = new Vue({
  el: '#v-for-workers-list',
  delimiters: ['[[', ']]'],
  data: {
    workers: []
  }
})

var name_of_node = new Vue({
  el: '#name_of_node',
  delimiters: ['[[', ']]'],
  data: {
    name: ""
  }
});

var online_status = new Vue({
  el: '#online_status',
  delimiters: ['[[', ']]'],
  data: {
    status: ""
  }
});

var server_ip = new Vue({
  el: '#server_ip',
  delimiters: ['[[', ']]'],
  data: {
    ip: baseUrl
  }
});

// MAIN LOGIC
async function update_server_status() {
  // just doing a identity check to see if server is online
  var identity = await get_status_from_server()
  if (identity["data"]["status"] == "OpenGrid") {
    set_online_status("Online")
  } else {
    set_online_status("Offline")
  }
}

async function update_name_of_node(){
  var identity = await get_identity_from_server()
  set_name_of_node(identity["data"]["identity"])
}
async function update_models_list() {
  var response = await get_models_from_server()
  set_models_in_table(response["data"]["models"])
}

async function update_workers_list() {
  var response = await get_workers_from_server()
  console.log(response)
  set_workers_in_table(response["data"]["workers"])
}

async function update_dataset_tags() {
  let response = await get_dataset_tags()
  console.log(response)
  set_tags_in_table(response["data"])
}

async function sync_with_server() {
  console.log("syncing with server")
  await update_server_status()
  await update_models_list()
  await update_name_of_node()
  await update_workers_list()
  await update_dataset_tags()
  setTimeout(sync_with_server, 5000)
}

sync_with_server()
