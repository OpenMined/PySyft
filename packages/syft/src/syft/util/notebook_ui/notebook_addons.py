# stdlib
import json
from string import Template

# relative
from ...types.uid import UID

CSS_CODE = """
<style>
  .header-1 {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 600;
        font-size: 2.0736em;
        line-height: 140%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

  .header-2 {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 600;
        font-size: 1.728em;
        line-height: 140%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

  .header-3 {
        font-family: Roboto';
        font-style: normal;
        font-weight: 600;
        font-size:  1.44em;
        line-height: 140%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }

  .header-4 {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 600;
        font-size: 1.2em;
        line-height: 140%;
        leading-trim: both;
        text-edge: cap;
        color: #17161D;
    }
    
    .paragraph {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 400;
        font-size: 14px;
        line-height: 140%;
        leading-trim: both;
        text-edge: cap;
        color: #2E2B3B;
    }
    
    .paragraph-sm {
        font-family: 'Roboto';
        font-style: normal;
        font-weight: 400;
        font-size: 11.62px;
        line-height: 140%;
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
    
    .repr-cell {
      padding-top: 56px;
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
    
    .pt-25{
        padding-top: 25px;
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
    
    .search-field{
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 8px 12px;
        width: 343px;
        height: 24px;
        /* Lt On Surface/Low */
        background: #F1F0F4;
        border-radius: 30px;

        /* Lt On Surface/Highest */
        color: #B4B0BF;
        border:none;
        /* Inside auto layout */
        flex: none;
        order: 0;
        flex-grow: 0;
    }
    .search-button{
        /* Search */
        leading-trim: both;
        text-edge: cap;
        display: flex;
        align-items: center;
        text-align: center;

        /* Primary/On Light */
        color: #464A91;

        border-radius: 30px;
        border-color: #464A91;
        /* Inside auto layout */
        flex: none;
        order: 1;
        flex-grow: 0;
    }
    
    .grid-table${uid} {
        display:grid;
        grid-template-columns: 1fr repeat(${cols}, 1fr);
        grid-template-rows: repeat(2, 1fr);
    }
    
    .grid-std-cells {
        grid-column: span 4;        
    }
    .grid-index-cells {
        grid-column: span 1;
    }
    
    .grid-header {
        /* Auto layout */
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 6px 4px;

        /* Lt On Surface/Surface */
        background: #ECEBEF;
        /* Lt On Surface/High */
        border: 1px solid #CFCDD6;
    }
    
    .grid-row {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 6px 4px;
        overflow: hidden;
        border: 1px solid #CFCDD6;
    }
    
    .badge {
        code-text;
        border-radius: 30px;
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
        height: 30px;
        display: flex;
        justify-content: center;
        gap: 8px;
        padding: 5px;
    }
    .page{
        color: black;
        font-weight: bold;
    }
    .page:hover {
      color: #38bdf8;
      cursor: pointer;
    }
</style>
"""


custom_code = """
    <div class='pt-25'>
        <div class='flex gap-10'>
            <div><svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"> <path d="M28 6H4C3.73478 6 3.48043 6.10536 3.29289 6.29289C3.10536 6.48043 3 6.73478 3 7V24C3 24.5304 3.21071 25.0391 3.58579 25.4142C3.96086 25.7893 4.46957 26 5 26H27C27.5304 26 28.0391 25.7893 28.4142 25.4142C28.7893 25.0391 29 24.5304 29 24V7C29 6.73478 28.8946 6.48043 28.7071 6.29289C28.5196 6.10536 28.2652 6 28 6ZM5 14H10V18H5V14ZM12 14H27V18H12V14ZM27 8V12H5V8H27ZM5 20H10V24H5V20ZM27 24H12V20H27V24Z" fill="#343330"/></svg></div>
                <div><p class='header-3'>${list_name}</p></div>
            </div>
            
            <div style="display:flex;justify-content: space-between; align-items: center;">
                <div class='pt-25 flex gap-10'>
                    <input id='searchKey${uid}' class='search-field' placeholder='Enter search here ...'  />
                    <button class='search-button'  onclick="searchGrid${uid}(element${uid})">
                        <svg width="11" height="10" viewBox="0 0 11 10" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M10.5652 9.23467L8.21819 6.88811C8.89846 6.07141 9.23767 5.02389 9.16527 3.96345C9.09287 2.90302 8.61443 1.91132 7.82948 1.19466C7.04453 0.477995 6.01349 0.0915414 4.95087 0.115691C3.88824 0.139841 2.87583 0.572735 2.12425 1.32432C1.37266 2.0759 0.939768 3.08831 0.915618 4.15094C0.891468 5.21357 1.27792 6.2446 1.99459 7.02955C2.71125 7.8145 3.70295 8.29294 4.76338 8.36535C5.82381 8.43775 6.87134 8.09853 7.68804 7.41827L10.0346 9.7653C10.0694 9.80014 10.1108 9.82778 10.1563 9.84663C10.2018 9.86549 10.2506 9.87519 10.2999 9.87519C10.3492 9.87519 10.398 9.86549 10.4435 9.84663C10.489 9.82778 10.5304 9.80014 10.5652 9.7653C10.6001 9.73046 10.6277 9.68909 10.6466 9.64357C10.6654 9.59805 10.6751 9.54926 10.6751 9.49998C10.6751 9.45071 10.6654 9.40192 10.6466 9.3564C10.6277 9.31088 10.6001 9.26951 10.5652 9.23467ZM1.67491 4.24998C1.67491 3.58247 1.87285 2.92995 2.2437 2.37493C2.61455 1.81992 3.14165 1.38734 3.75835 1.13189C4.37506 0.876446 5.05366 0.809609 5.70834 0.939835C6.36303 1.07006 6.96439 1.3915 7.4364 1.8635C7.9084 2.3355 8.22984 2.93687 8.36006 3.59155C8.49029 4.24624 8.42345 4.92484 8.168 5.54154C7.91256 6.15824 7.47998 6.68535 6.92496 7.05619C6.36995 7.42704 5.71742 7.62498 5.04991 7.62498C4.15511 7.62399 3.29724 7.26809 2.66452 6.63537C2.0318 6.00265 1.6759 5.14479 1.67491 4.24998Z" fill="#464A91"/></svg>
                        <span class='pl-8'>Search</span>
                    </button>
                </div>
                
                <div><h4 id='total${uid}'>0</h4></div>
            </div>
            <div id='table${uid}' class='grid-table${uid}' style='margin-top: 25px;'>
                <script>
                    var element${uid} = ${element}
                    var page_size${uid} = ${rows}
                    var pageIndex${uid} = 1
                    var paginatedElements${uid} = []


                    function paginate${uid}(arr, size) {
                        const res = [];
                        for (let i = 0; i < arr.length; i += size) {
                            const chunk = arr.slice(i, i + size);
                            res.push(chunk);
                        }

                        return res;
                    }

                    function searchGrid${uid}(elements){
                        let searchKey = document.getElementById('searchKey${uid}').value
                        let result = elements.filter((element) => { 
                            return element.id.includes(searchKey)
                        } );
                        resetById${uid}('table${uid}');
                        resetById${uid}('pag${uid}');
                        pageIndex = 1
                        result = paginate${uid}(result, page_size${uid})
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
                                    let table_index${uid} = 0
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
                                            if (typeof item[attr] === 'object' && item[attr] !== null && item[attr].hasOwnProperty('type')){
                                                if (item[attr].type.includes('badge')){
                                                    let badge_div = document.createElement("div");
                                                    badge_div.classList.add('badge',item[attr].type)
                                                    badge_div.innerText = item[attr].value.toUpperCase();
                                                    div.appendChild(badge_div);
                                                    div.classList.add('grid-row','grid-std-cells');
                                                    div.style.justifyContent = 'center';
                                                }
                                            } else{
                                                div.classList.add('grid-row','grid-std-cells');
                                                let text = ''
                                                if (attr === 'id' ){
                                                    text = item[attr].slice(0, 8) + "..." + item[attr].slice(-3);
                                                    div.onclick = function() { 
                                                        navigator.clipboard.writeText(item[attr]); 
                                                    };
                                                } 
                                                else if (item[attr] == null) {
                                                    text = ' '
                                                } else {
                                                    text = item[attr]
                                                }
                                                div.innerText = text;
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
                                  div.style.color = 'gray';
                                  div.onclick = function(event) {
                                      let indexes = document.getElementsByClassName('page');
                                      for (let index of indexes) { index.style.color = 'gray' }
                                      event.target.style.color = "black"
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

                    buildPaginationContainer${uid}(paginatedElements${uid})
                </script>
            </div>
        </div>
    </div>
"""


def create_table_template(items, list_name, rows=5):
    items_dict = json.dumps(items)
    code = CSS_CODE + custom_code
    template = Template(code)
    cols = (len(items[0].keys())) * 4
    return template.substitute(
        uid=str(UID()), element=items_dict, list_name=list_name, cols=cols, rows=rows
    )
