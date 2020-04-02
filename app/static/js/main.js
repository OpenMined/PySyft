var getUrl = window.location;
var baseUrl = getUrl.protocol + "//" + getUrl.host + "/" + getUrl.pathname.split('/')[1];

var server_ip = new Vue({
    el: '#server_ip',
    delimiters: ['[[', ']]'],
    data: {
        ip: baseUrl
    }
});