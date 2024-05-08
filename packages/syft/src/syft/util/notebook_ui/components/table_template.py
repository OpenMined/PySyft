# stdlib
from collections.abc import Sequence
import json
from string import Template

# relative
from ....types.uid import UID
from ..icons import Icon
from ..styles import CSS_CODE

TABLE_INDEX_KEY = "_table_repr_index"

custom_code = """
<style>
    /* TODO Refactor table and remove templated CSS classes */
    .grid-table${uid} {
        display:grid;
        grid-template-columns: ${grid_template_columns};
        /*grid-template-rows: repeat(2, 1fr);*/
        position: relative;
    }

    .grid-std-cells${uid} {
        grid-column: ${grid_template_cell_columns};
        display: flex;
        justify-content: center;
        align-items: center;
    }
</style>

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
                                // remove index from header
                                headers = headers.filter((header) => header !== '_table_repr_index');

                                let grid = document.getElementById("table${uid}");
                                let div = document.createElement("div");
                                div.classList.add('grid-header', 'grid-index-cells');
                                grid.appendChild(div);
                                headers.forEach((title) =>{
                                    let div = document.createElement("div");
                                    div.classList.add('grid-header', 'grid-std-cells${uid}');
                                    div.innerText = title;

                                    grid.appendChild(div);
                                });

                                let page = items[pageIndex -1]
                                if (page !== 'undefined'){
                                    let table_index${uid} = ((pageIndex - 1) * page_size${uid});
                                    page.forEach((item) => {
                                        let grid = document.getElementById("table${uid}");
                                        // Add new index value in index cells
                                        let divIndex = document.createElement("div");
                                        divIndex.classList.add('grid-row', 'grid-index-cells');
                                        let itemIndex;
                                        if ('_table_repr_index' in item) {
                                            itemIndex = item['_table_repr_index'];
                                        } else {
                                            itemIndex = table_index${uid};
                                        }
                                        divIndex.innerText = itemIndex;
                                        grid.appendChild(divIndex);

                                        // Iterate over the actual obj
                                        for (const attr in item) {
                                            if (attr === '_table_repr_index') continue;

                                            let div = document.createElement("div");
                                            if (typeof item[attr] === 'object'
                                                && item[attr] !== null
                                                && item[attr].hasOwnProperty('type')) {
                                                if (item[attr].type.includes('badge')){
                                                    let badge_div = document.createElement("div");
                                                    badge_div.classList.add('badge',item[attr].type)
                                                    badge_div.innerText = String(item[attr].value).toUpperCase();
                                                    div.appendChild(badge_div);
                                                    div.classList.add('grid-row','grid-std-cells${uid}');
                                                } else if (item[attr].type.includes('label')){
                                                    let label_div = document.createElement("div");
                                                    label_div.classList.add('label',item[attr].type)
                                                    label_div.innerText = String(item[attr].value).toUpperCase();
                                                    label_div.classList.add('center-content-cell');
                                                    div.appendChild(label_div);
                                                    div.classList.add('grid-row','grid-std-cells${uid}');
                                                } else if (item[attr].type === "clipboard") {
                                                    div.classList.add('grid-row','grid-std-cells${uid}');

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
                                                    clipboard_img.innerHTML = ${clipboardIconEscaped};

                                                    clipboard_div.appendChild(clipboard_img);
                                                    div.appendChild(clipboard_div);
                                                }
                                            } else{
                                                div.classList.add('grid-row','grid-std-cells${uid}');
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
                        const myFont = new FontFace('DejaVu Sans', 'url(https://cdn.jsdelivr.net/npm/dejavu-sans@1.0.0/fonts/dejavu-sans-webfont.woff2?display=swap');
                        await myFont.load();
                        document.fonts.add(myFont);
                    })();

                    buildPaginationContainer${uid}(paginatedElements${uid})
                </script>
            </div>
        </div>
"""


def create_table_template(
    table_data: Sequence,
    name: str,
    rows: int = 5,
    icon: str | None = None,
    grid_template_columns: str | None = None,
    grid_template_cell_columns: str | None = None,
    **kwargs: dict,
) -> str:
    if icon is None:
        icon = Icon.TABLE.svg
    if grid_template_columns is None:
        grid_template_columns = "1fr repeat({cols}, 1fr)"
    if grid_template_cell_columns is None:
        grid_template_cell_columns = "span 4"

    items_dict = json.dumps(table_data)
    code = CSS_CODE + custom_code
    template = Template(code)
    rows = min(len(table_data), rows)
    if len(table_data) == 0:
        cols = 0
    else:
        col_names = [k for k in table_data[0].keys() if k != TABLE_INDEX_KEY]
        cols = (len(col_names)) * 4
    if "{cols}" in grid_template_columns:
        grid_template_columns = grid_template_columns.format(cols=cols)
    final_html = template.substitute(
        uid=str(UID()),
        element=items_dict,
        list_name=name,
        cols=cols,
        rows=rows,
        icon=icon,
        searchIcon=Icon.SEARCH.svg,
        clipboardIconEscaped=Icon.CLIPBOARD.js_escaped_svg,
        grid_template_columns=grid_template_columns,
        grid_template_cell_columns=grid_template_cell_columns,
    )
    return final_html
