# stdlib
from collections.abc import Sequence
import json
from string import Template
from typing import Any

# relative
from ...types.uid import UID
from .icons import Icon

STYLESHEET_URLS = [
    "https://fonts.googleapis.com/css2?family=Karla:ital,wght@0,200;0,300;0,400;0,500;0,600;0,700;0,800;1,200;1,300;1,400;1,500;1,600;1,700;1,800&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap",
    "https://fonts.cdnfonts.com/css/dejavu-sans-mono",
]
STYLESHEET_JS_CALLS = "\n".join([f'addStyleSheet("{s}")' for s in STYLESHEET_URLS])


JS_DOWNLOAD_FONTS = f"""
<script>
function addStyleSheet(fileName) {{

  var head = document.head;
  var link = document.createElement("link");

  link.type = "text/css";
  link.rel = "stylesheet";
  link.href = fileName;

  head.appendChild(link);
}}

{STYLESHEET_JS_CALLS}
</script>
"""


CSS = """
<style>
  .syft-widget body[data-jp-theme-light='false'] {
        --primary-color: #111111;
        --secondary-color: #212121;
        --tertiary-color: #CFCDD6;
        --button-color: #111111;
  }

  .syft-widget body {
        --primary-color: #ffffff;
        --secondary-color: #f5f5f5;
        --tertiary-color: #000000de;
        --button-color: #d1d5db;
  }

  .header-1 {
        font-style: normal;
        font-weight: 600;
        font-size: 2.0736em;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

  .header-2 {
        font-style: normal;
        font-weight: 600;
        font-size: 1.728em;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

  .header-3 {
        font-style: normal;
        font-weight: 600;
        font-size:  1.44em;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: var(--tertiary-color);
    }

  .header-4 {
        font-style: normal;
        font-weight: 600;
        font-size: 1.2em;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

    .paragraph {
        font-style: normal;
        font-weight: 400;
        font-size: 14px;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: #2E2B3B;
    }

    .paragraph-sm {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 400;
        font-size: 11.62px;
        line-height: 100%;
        leading-trim: both;
        text-edge: cap;
        color: #2E2B3B;
    }
    .code-text {
        font-family: 'Consolas';
        font-style: normal;
        font-weight: 400;
        font-size: 13px;
        line-height: 130%;
        leading-trim: both;
        text-edge: cap;
        color: #2E2B3B;
    }

    .numbering-entry { display: none }

    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black; /* If you want dots under the hoverable text */
    }

    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        padding: 5px 0;
        border-radius: 6px;

        /* Position the tooltip text - see examples below! */
        position: absolute;
        z-index: 1;
    }

    .repr-cell {
      padding-top: 20px;
    }

    .text-bold {
        font-weight: bold;
    }

    .pr-8 {
        padding-right: 8px;
    }
    .pt-8 {
        padding-top: 8px;
    }
    .pl-8 {
        padding-left: 8px;
    }
    .pb-8 {
        padding-bottom: 8px;
    }

    .py-25{
        padding-top: 25px;
        padding-bottom: 25px;
    }

    .flex {
        display: flex;
    }

    .gap-10 {
        gap: 10px;
    }
    .items-center{
        align-items: center;
    }

    .folder-icon {
        color: var(--tertiary-color);
    }

    .search-input{
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 8px 12px;
        width: 343px;
        height: 24px;
        /* Lt On Surface/Low */
        background-color: var(--secondary-color);
        border-radius: 30px;

        /* Lt On Surface/Highest */
        color: var(--tertiary-color);
        border:none;
        /* Inside auto layout */
        flex: none;
        order: 0;
        flex-grow: 0;
    }
    .search-input:focus {
        outline: none;
    }
        .search-input:focus::placeholder,
    .search-input::placeholder { /* Chrome, Firefox, Opera, Safari 10.1+ */
        color: var(--tertiary-color);
        opacity: 1; /* Firefox */
    }

    .search-button{
        /* Search */
        leading-trim: both;
        text-edge: cap;
        display: flex;
        align-items: center;
        text-align: center;

        /* Primary/On Light */
        background-color: var(--button-color);
        color: var(--tertiary-color);

        border-radius: 30px;
        border-color: var(--secondary-color);
        border-style: solid;
        box-shadow: rgba(60, 64, 67, 0.3) 0px 1px 2px 0px, rgba(60, 64, 67, 0.15) 0px 1px 3px 1px;
        cursor: pointer;
        /* Inside auto layout */
        flex: none;
        order: 1;
        flex-grow: 0;
    }

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

    .grid-index-cells {
        grid-column: span 1;
        /* tmp fix to make left col stand out (fix with font-family) */
        font-weight: 600;
        background-color: var(--secondary-color) !important;
        color: var(--tertiary-color);
    }

    .center-content-cell{
        margin: auto;
    }

    .grid-header {
        /* Auto layout */
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 6px 4px;

        resize: horizontal;
        /* Lt On Surface/Surface */
        /* Lt On Surface/High */
        border: 1px solid #CFCDD6;
        /* tmp fix to make header stand out (fix with font-family) */
        font-weight: 600;
        background-color: var(--secondary-color);
        color: var(--tertiary-color);
    }

    .grid-row {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 6px 4px;
        overflow: hidden;
        border: 1px solid #CFCDD6;
        background-color: var(--primary-color);
        color: var(--tertiary-color);
    }


    .syncstate-col-footer {
        font-family: 'DejaVu Sans Mono', 'Open Sans';
        font-size: 12px;
        font-weight: 400;
        line-height: 16.8px;
        text-align: left;
        color: #5E5A72;
    }

    .syncstate-description {
        font-family: Open Sans;
        font-size: 14px;
        font-weight: 600;
        line-height: 19.6px;
        text-align: left;
        white-space: nowrap;
        flex-grow: 1;
    }

    .widget-header2{
        display: flex;
        gap: 8px;
        justify-content: start;
        width: 100%;
        overflow: hidden;
        align-items: center;
    }

    .widget-header2-2{
        display: flex;
        gap: 8px;
        justify-content: start;
        align-items: center;
    }

    .jobs-title {
        font-family: Open Sans, sans-serif;
        font-size: 18px;
        font-weight: 600;
        line-height: 25.2px;
        text-align: left;
        color: #1F567A;
    }

    .diff-state-orange-text{
        color: #B8520A;
    }

    .diff-state-no-obj{
        font-family: 'DejaVu Sans Mono', 'Open Sans';
        font-size: 12px;
        font-weight: 400;
        line-height: 16.8px;
        text-align: left;
        color: #5E5A72;
    }

    .diff-state-intro{
        font-family: Open Sans;
        font-size: 14px;
        font-weight: 400;
        line-height: 19.6px;
        text-align: left;
        color: #B4B0BF;
    }

    .diff-state-header{
        font-family: Open Sans;
        font-size: 22px;
        font-weight: 600;
        line-height: 30.8px;
        text-align: left;
        color: #353243;
        display: flex; gap: 8px;
    }

    .diff-state-sub-header{
        font-family: Open Sans;
        font-size: 14px;
        font-weight: 400;
        line-height: 19.6px;
        text-align: left;
        color: #5E5A72;
    }

    .badge {
        code-text;
        border-radius: 30px;
    }

    .label {
        code-text;
        border-radius: 4px;
        padding: 6px 4px;
        white-space: nowrap;
        overflow: hidden;
        line-height: 1.2;
        font-family: monospace;
    }

    .label-light-purple {
        label;
        background-color: #C9CFE8;
        color: #373B7B;
    }

    .label-light-blue {
        label;
        background-color: #C2DEF0;
        color: #1F567A;
    }

    .label-orange {
        badge;
        background-color: #FEE9CD;
        color: #B8520A;
    }

    .label-gray {
        badge;
        background-color: #ECEBEF;
        color: #353243;
    }

    .label-green {
        badge;
        background-color: #D5F1D5;
        color: #256B24;
    }

    .label-red {
        label;
        background-color: #F2D9DE;
        color: #9B2737;
    }

    .badge-blue {
        badge;
        background-color: #C2DEF0;
        color: #1F567A;
    }

    .badge-purple {
        badge;
        background-color: #C9CFE8;
        color: #373B7B;
    }

    .badge-green {
        badge;

        /* Success/Container */
        background-color: #D5F1D5;
        color: #256B24;
    }

    .badge-red {
        badge;
        background-color: #F2D9DE;
        color: #9B2737;
    }

    .badge-gray {
        badge;
        background-color: #ECEBEF;
        color: #2E2B3B;
    }
    .paginationContainer{
        width: 100%;
        /*height: 30px;*/
        display: flex;
        justify-content: center;
        gap: 8px;
        padding: 5px;
        color: var(--tertiary-color);
    }

    .widget-label-basic{
        display:flex;
    }

    .widget-label-basic input[type='checkbox'][disabled] {
        filter: sepia(0.3) hue-rotate(67deg) saturate(3);
    }

    .page{
        color: black;
        font-weight: bold;
        color: var(--tertiary-color);
    }
    .page:hover {
      color: #38bdf8;
      cursor: pointer;
    }
    .clipboard:hover{
        cursor: pointer;
        color: var(--tertiary-color);
    }

    .rendered_html tbody tr:nth-child(odd) {
        background: transparent;
    }

    .search-field {
        display: flex;
        align-items: center;
        border-radius: 30px;
        background-color: var(--secondary-color);
    }

    .syft-dropdown {
        margin: 5px;
        margin-left: 5px;
        position: relative;
        display: inline-block;
        text-align: center;
        background-color: var(--button-color);
        min-width: 100px;
        padding: 2px;
        border-radius: 30px;
    }

    .syft-dropdown:hover {
        cursor: pointer;
    }
    .syft-dropdown-content {
        margin-top:26px;
        display: none;
        position: absolute;
        min-width: 100px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        padding: 12px 6px;
        z-index: 1;
        background-color: var(--primary-color);
        color: var(--tertiary-color);
    }
    .dd-options {
        padding-top: 4px;
    }
    .dd-options:first-of-type {
        padding-top: 0px;
    }

    .dd-options:hover {
        cursor: pointer;
        background: #d1d5db;
    }
    .arrow {
        border: solid black;
        border-width: 0 3px 3px 0;
        display: inline-block;
        padding: 3px;
    }
    .down {
        transform: rotate(45deg);
        -webkit-transform: rotate(45deg);
    }
</style>

"""

CSS_CODE = f"""
{CSS}
"""

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
                                    div.classList.add('grid-header', 'grid-std-cells${uid}');
                                    div.innerText = title;

                                    grid.appendChild(div);
                                });

                                let page = items[pageIndex -1]
                                if (page !== 'undefined'){
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
                                                    clipboard_img.innerHTML = "${clipboardIcon}";

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
    items: Sequence,
    list_name: str,
    rows: int = 5,
    table_icon: str | None = None,
    grid_template_columns: str | None = None,
    grid_template_cell_columns: str | None = None,
) -> str:
    if table_icon is None:
        table_icon = Icon.TABLE.svg
    if grid_template_columns is None:
        grid_template_columns = "1fr repeat({cols}, 1fr)"
    if grid_template_cell_columns is None:
        grid_template_cell_columns = "span 4"

    items_dict = json.dumps(items)
    code = CSS_CODE + custom_code
    template = Template(code)
    rows = min(len(items), rows)
    if len(items) == 0:
        cols = 0
    else:
        cols = (len(items[0].keys())) * 4
    if "{cols}" in grid_template_columns:
        grid_template_columns = grid_template_columns.format(cols=cols)
    final_html = template.substitute(
        uid=str(UID()),
        element=items_dict,
        list_name=list_name,
        cols=cols,
        rows=rows,
        icon=table_icon,
        searchIcon=Icon.SEARCH.svg,
        clipboardIcon=Icon.CLIPBOARD.svg,
        grid_template_columns=grid_template_columns,
        grid_template_cell_columns=grid_template_cell_columns,
    )
    return final_html
