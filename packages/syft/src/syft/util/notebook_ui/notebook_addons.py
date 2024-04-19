# stdlib
from collections.abc import Sequence
import json
from string import Template
from typing import Any

# relative
from ...types.uid import UID
from ..resources import read_css
from ..resources import read_svg

CSS_CODE = f"""
<style>
{read_css("style.css")}
</style>
"""

SEARCH_ICON = read_svg("search.svg")
CLIPBOARD_ICON = read_svg("clipboard.svg")
TABLE_ICON = read_svg("table.svg")
FOLDER_ICON = read_svg("folder.svg")
REQUEST_ICON = read_svg("request.svg")
ARROW_ICON = read_svg("arrow.svg")

custom_code = """
    <div style='margin-top:15px;'>
        <div class='flex gap-10' style='align-items: center;'>
            <div class='folder-icon'>${icon}</div>
            <div><p class='header-3'>${list_name}</p></div>
        </div>

            <div style="padding-top: 16px; display:flex;justify-content: space-between; align-items: center;">
                <div class='pt-25 gap-10' style="display:flex;">
                    <div class="search-field">
                        <div id='search-menu${uid}' class="syft-dropdown" onclick="{
                            let doc = document.getElementById('search-dropdown-content${uid}')
                            if (doc.style.display === 'block'){
                                doc.style.display = 'none'
                            } else {
                                doc.style.display = 'block'
                            }
                            }">
                            <div id='search-dropdown-content${uid}' class='syft-dropdown-content'></div>
                            <script>
                                var element${uid} = ${element}
                                var page_size${uid} = ${rows}
                                var pageIndex${uid} = 1
                                var paginatedElements${uid} = []
                                var activeFilter${uid};

                                function buildDropDownMenu(elements){
                                    let init_filter;
                                    let menu = document.getElementById('search-dropdown-content${uid}')
                                    if (elements.length > 0) {
                                        let sample = elements[0]
                                        for (const attr in sample) {
                                            if (typeof init_filter === 'undefined'){
                                                init_filter = attr;
                                            }
                                            let content = document.createElement('div');
                                            content.onclick = function(event) {
                                                event.stopPropagation()
                                                document.getElementById('menu-active-filter${uid}').innerText = attr;
                                                activeFilter${uid} = attr;
                                                document.getElementById(
                                                    'search-dropdown-content${uid}'
                                                ).style.display= 'none';
                                            }
                                            content.classList.add("dd-options");
                                            content.innerText = attr;
                                            menu.appendChild(content);
                                        }
                                    } else {
                                        let init_filter = '---'
                                    }
                                    let dropdown_field = document.getElementById('search-menu${uid}')
                                    let span = document.createElement('span')
                                    span.setAttribute('id', 'menu-active-filter${uid}')
                                    span.innerText = init_filter
                                    activeFilter${uid} = init_filter;
                                    dropdown_field.appendChild(span)
                                }

                                buildDropDownMenu(element${uid})
                            </script>
                        </div>
                        <input id='searchKey${uid}' class='search-input' placeholder='Enter search here ...'  />
                    </div>
                    <button class='search-button' type="button" onclick="searchGrid${uid}(element${uid})">
                        ${searchIcon}
                        <span class='pl-8'>Search</span>
                    </button>
                </div>

                <div><h4 id='total${uid}'>0</h4></div>
            </div>
            <div id='table${uid}' class='grid-table${uid}' style='margin-top: 25px;'>
                <script>
                    function paginate${uid}(arr, size) {
                        const res = [];
                        for (let i = 0; i < arr.length; i += size) {
                            const chunk = arr.slice(i, i + size);
                            res.push(chunk);
                        }

                        return res;
                    }

                    function searchGrid${uid}(elements){
                        let searchKey = document.getElementById('searchKey${uid}').value;
                        let result;
                        if (searchKey === ''){
                            result = elements;
                        } else {
                            result = elements.filter((element) => {
                                let property = element[activeFilter${uid}]
                                if (typeof property === 'object' && property !== null){
                                    return property.value.toLowerCase().includes(searchKey.toLowerCase());
                                } else if (typeof property === 'string' ) {
                                    return element[activeFilter${uid}].toLowerCase().includes(searchKey.toLowerCase());
                                } else if (property !== null ) {
                                    return element[activeFilter${uid}].toString() === searchKey;
                                } else {
                                    return element[activeFilter${uid}] === searchKey;
                                }
                            } );
                        }
                        resetById${uid}('table${uid}');
                        resetById${uid}('pag${uid}');
                        result = paginate${uid}(result, page_size${uid})
                        paginatedElements${uid} = result
                        buildGrid${uid}(result,pageIndex${uid});
                        buildPaginationContainer${uid}(result);
                    }

                    function resetById${uid}(id){
                        let element = document.getElementById(id);
                        while (element.firstChild) {
                          element.removeChild(element.firstChild);
                        }
                    }

                    function buildGrid${uid}(items, pageIndex){
                                let headers = Object.keys(element${uid}[0]);

                                let grid = document.getElementById("table${uid}");
                                let div = document.createElement("div");
                                div.classList.add('grid-header', 'grid-index-cells');
                                grid.appendChild(div);
                                headers.forEach((title) =>{
                                    let div = document.createElement("div");
                                    div.classList.add('grid-header', 'grid-std-cells');
                                    div.innerText = title;

                                    grid.appendChild(div);
                                });

                                let page = items[pageIndex -1]
                                if (page !== 'undefine'){
                                    let table_index${uid} = ((pageIndex - 1) * page_size${uid})
                                    page.forEach((item) => {
                                        let grid = document.getElementById("table${uid}");
                                        // Add new index value in index cells
                                        let divIndex = document.createElement("div");
                                        divIndex.classList.add('grid-row', 'grid-index-cells');
                                        divIndex.innerText = table_index${uid};
                                        grid.appendChild(divIndex);

                                        // Iterate over the actual obj
                                        for (const attr in item) {
                                            let div = document.createElement("div");
                                            if (typeof item[attr] === 'object'
                                                && item[attr] !== null
                                                && item[attr].hasOwnProperty('type')) {
                                                if (item[attr].type.includes('badge')){
                                                    let badge_div = document.createElement("div");
                                                    badge_div.classList.add('badge',item[attr].type)
                                                    badge_div.innerText = String(item[attr].value).toUpperCase();
                                                    div.appendChild(badge_div);
                                                    div.classList.add('grid-row','grid-std-cells');
                                                } else if (item[attr].type.includes('label')){
                                                    let label_div = document.createElement("div");
                                                    label_div.classList.add('label',item[attr].type)
                                                    label_div.innerText = String(item[attr].value).toUpperCase();
                                                    div.appendChild(label_div);
                                                    div.classList.add('grid-row','grid-std-cells');
                                                } else if (item[attr].type === "clipboard") {
                                                    div.classList.add('grid-row','grid-std-cells');

                                                    // Create clipboard div
                                                    let clipboard_div = document.createElement('div');
                                                    clipboard_div.style.display= 'flex';
                                                    clipboard_div.classList.add("gap-10")
                                                    clipboard_div.style.justifyContent = "space-between";

                                                    let id_text = document.createElement('div');
                                                    if (item[attr].value == "None"){
                                                        id_text.innerText = "None";
                                                    }
                                                    else{
                                                        id_text.innerText = item[attr].value.slice(0,5) + "...";
                                                    }

                                                    clipboard_div.appendChild(id_text);
                                                    let clipboard_img = document.createElement('div');
                                                    clipboard_img.classList.add("clipboard")
                                                    div.onclick = function() {
                                                        navigator.clipboard.writeText(item[attr].value);
                                                    };
                                                    clipboard_img.innerHTML = "${clipboardIcon}";

                                                    clipboard_div.appendChild(clipboard_img);
                                                    div.appendChild(clipboard_div);
                                                }
                                            } else{
                                                div.classList.add('grid-row','grid-std-cells');
                                                if (item[attr] == null) {
                                                    text = ' '
                                                } else {
                                                    text = String(item[attr])
                                                }

                                                text = text.replaceAll("\\n", "</br>");
                                                div.innerHTML = text;
                                            }
                                            grid.appendChild(div);
                                        }
                                    table_index${uid} = table_index${uid} + 1;
                                    })
                                }
                    }
                    paginatedElements${uid} = paginate${uid}(element${uid}, page_size${uid})
                    buildGrid${uid}(paginatedElements${uid}, 1)
                    document.getElementById('total${uid}').innerText = "Total: " + element${uid}.length
                </script>
            </div>
            <div id='pag${uid}' class='paginationContainer'>
                <script>
                    function buildPaginationContainer${uid}(paginatedElements){
                            let pageContainer = document.getElementById("pag${uid}");
                            for (let i = 0; i < paginatedElements.length; i++) {
                                  let div = document.createElement("div");
                                  div.classList.add('page');
                                  if(i===0) div.style.color = "gray";
                                  else div.style.color = 'var(--tertiary-color, "gray")';
                                  div.onclick = function(event) {
                                      let indexes = document.getElementsByClassName('page');
                                      for (let index of indexes) { index.style.color = 'var(--tertiary-color, "gray")' }
                                      event.target.style.color = "gray";
                                      setPage${uid}(i + 1);
                                  };
                                  div.innerText = i + 1;
                                  pageContainer.appendChild(div);
                            }
                    }

                    function setPage${uid}(newPage){
                        pageIndex = newPage
                        resetById${uid}('table${uid}')
                        buildGrid${uid}(paginatedElements${uid}, pageIndex)
                    }
                    (async function() {
                        const myFont = new FontFace('DejaVu Sans', 'url(https://cdn.jsdelivr.net/npm/dejavu-sans@1.0.0/css/dejavu-sans.min.css)');
                        await myFont.load();
                        document.fonts.add(myFont);
                        document.getElementsByTagName('h1')[0].style.fontFamily = "DejaVu Sans";
                    })();

                    buildPaginationContainer${uid}(paginatedElements${uid})
                </script>
            </div>
        </div>
"""


def create_table_template(
    items: Sequence, list_name: Any, rows: int = 5, table_icon: Any = None
) -> str:
    if not table_icon:
        table_icon = TABLE_ICON

    items_dict = json.dumps(items)
    code = CSS_CODE + custom_code
    template = Template(code)
    rows = min(len(items), rows)
    if len(items) == 0:
        cols = 0
    else:
        cols = (len(items[0].keys())) * 4
    return template.substitute(
        uid=str(UID()),
        element=items_dict,
        list_name=list_name,
        cols=cols,
        rows=rows,
        icon=table_icon,
        searchIcon=SEARCH_ICON,
        clipboardIcon=CLIPBOARD_ICON,
    )
