# stdlib
from collections.abc import Sequence
import json
from string import Template
from typing import Any

# relative
from ...types.uid import UID

CSS_CODE = """
<style>
  body[data-jp-theme-light='false'] {
        --primary-color: #111111;
        --secondary-color: #212121;
        --tertiary-color: #CFCDD6;
        --button-color: #111111;
  }

  body {
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
        grid-template-columns: 1fr repeat(${cols}, 1fr);
        grid-template-rows: repeat(2, 1fr);
        overflow-x: auto;
        position: relative;
    }

    .grid-std-cells {
        grid-column: span 4;

    }
    .grid-index-cells {
        grid-column: span 1;
        /* tmp fix to make left col stand out (fix with font-family) */
        font-weight: 600;
        background-color: var(--secondary-color) !important;
        color: var(--tertiary-color);
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
        padding: 0px 4px;

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

SEARCH_ICON = (
    '<svg width="11" height="10" viewBox="0 0 11 10" fill="none"'
    ' xmlns="http://www.w3.org/2000/svg"><path d="M10.5652 9.23467L8.21819'
    " 6.88811C8.89846 6.07141 9.23767 5.02389 9.16527 3.96345C9.09287 2.90302 8.61443"
    " 1.91132 7.82948 1.19466C7.04453 0.477995 6.01349 0.0915414 4.95087"
    " 0.115691C3.88824 0.139841 2.87583 0.572735 2.12425 1.32432C1.37266 2.0759"
    " 0.939768 3.08831 0.915618 4.15094C0.891468 5.21357 1.27792 6.2446 1.99459"
    " 7.02955C2.71125 7.8145 3.70295 8.29294 4.76338 8.36535C5.82381 8.43775 6.87134"
    " 8.09853 7.68804 7.41827L10.0346 9.7653C10.0694 9.80014 10.1108 9.82778 10.1563"
    " 9.84663C10.2018 9.86549 10.2506 9.87519 10.2999 9.87519C10.3492 9.87519 10.398"
    " 9.86549 10.4435 9.84663C10.489 9.82778 10.5304 9.80014 10.5652 9.7653C10.6001"
    " 9.73046 10.6277 9.68909 10.6466 9.64357C10.6654 9.59805 10.6751 9.54926 10.6751"
    " 9.49998C10.6751 9.45071 10.6654 9.40192 10.6466 9.3564C10.6277 9.31088 10.6001"
    " 9.26951 10.5652 9.23467ZM1.67491 4.24998C1.67491 3.58247 1.87285 2.92995 2.2437"
    " 2.37493C2.61455 1.81992 3.14165 1.38734 3.75835 1.13189C4.37506 0.876446 5.05366"
    " 0.809609 5.70834 0.939835C6.36303 1.07006 6.96439 1.3915 7.4364 1.8635C7.9084"
    " 2.3355 8.22984 2.93687 8.36006 3.59155C8.49029 4.24624 8.42345 4.92484 8.168"
    " 5.54154C7.91256 6.15824 7.47998 6.68535 6.92496 7.05619C6.36995 7.42704 5.71742"
    " 7.62498 5.04991 7.62498C4.15511 7.62399 3.29724 7.26809 2.66452 6.63537C2.0318"
    ' 6.00265 1.6759 5.14479 1.67491 4.24998Z" fill="currentColor"/></svg>'
)
CLIPBOARD_ICON = (
    "<svg width='8' height='8' viewBox='0 0 8 8' fill='none'"
    " xmlns='http://www.w3.org/2000/svg'><path d='M7.4375 0.25H2.4375C2.35462 0.25"
    " 2.27513 0.282924 2.21653 0.341529C2.15792 0.400134 2.125 0.47962 2.125"
    " 0.5625V2.125H0.5625C0.47962 2.125 0.400134 2.15792 0.341529 2.21653C0.282924"
    " 2.27513 0.25 2.35462 0.25 2.4375V7.4375C0.25 7.52038 0.282924 7.59987 0.341529"
    " 7.65847C0.400134 7.71708 0.47962 7.75 0.5625 7.75H5.5625C5.64538 7.75 5.72487"
    " 7.71708 5.78347 7.65847C5.84208 7.59987 5.875 7.52038 5.875"
    " 7.4375V5.875H7.4375C7.52038 5.875 7.59987 5.84208 7.65847 5.78347C7.71708 5.72487"
    " 7.75 5.64538 7.75 5.5625V0.5625C7.75 0.47962 7.71708 0.400134 7.65847"
    " 0.341529C7.59987 0.282924 7.52038 0.25 7.4375 0.25ZM5.25"
    " 7.125H0.875V2.75H5.25V7.125ZM7.125 5.25H5.875V2.4375C5.875 2.35462 5.84208"
    " 2.27513 5.78347 2.21653C5.72487 2.15792 5.64538 2.125 5.5625"
    " 2.125H2.75V0.875H7.125V5.25Z' fill='#464158'/></svg>"
)
TABLE_ICON = (
    '<svg width="32" height="32" viewBox="0 0 32 32" fill="none"'
    ' xmlns="http://www.w3.org/2000/svg"> <path d="M28 6H4C3.73478 6 3.48043 6.10536'
    " 3.29289 6.29289C3.10536 6.48043 3 6.73478 3 7V24C3 24.5304 3.21071 25.0391"
    " 3.58579 25.4142C3.96086 25.7893 4.46957 26 5 26H27C27.5304 26 28.0391 25.7893"
    " 28.4142 25.4142C28.7893 25.0391 29 24.5304 29 24V7C29 6.73478 28.8946 6.48043"
    " 28.7071 6.29289C28.5196 6.10536 28.2652 6 28 6ZM5 14H10V18H5V14ZM12"
    ' 14H27V18H12V14ZM27 8V12H5V8H27ZM5 20H10V24H5V20ZM27 24H12V20H27V24Z"'
    ' fill="#343330"/></svg>'
)
FOLDER_ICON = (
    '<svg width="32"  height="32" viewBox="0 0 14 12" fill="none"'
    ' xmlns="http://www.w3.org/2000/svg"><path d="M13 2H8.66687L6.93313 0.7C6.75978'
    " 0.57066 6.54941 0.500536 6.33313 0.5H3.5C3.23478 0.5 2.98043 0.605357 2.79289"
    " 0.792893C2.60536 0.98043 2.5 1.23478 2.5 1.5V2.5H1.5C1.23478 2.5 0.98043 2.60536"
    " 0.792893 2.79289C0.605357 2.98043 0.5 3.23478 0.5 3.5V10.5C0.5 10.7652 0.605357"
    " 11.0196 0.792893 11.2071C0.98043 11.3946 1.23478 11.5 1.5 11.5H11.0556C11.306"
    " 11.4997 11.546 11.4001 11.723 11.223C11.9001 11.046 11.9997 10.806 12"
    " 10.5556V9.5H13.0556C13.306 9.49967 13.546 9.40007 13.723 9.22303C13.9001 9.046"
    " 13.9997 8.80599 14 8.55562V3C14 2.73478 13.8946 2.48043 13.7071 2.29289C13.5196"
    " 2.10536 13.2652 2 13 2ZM11 10.5H1.5V3.5H4.33313L6.06687 4.8C6.24022 4.92934"
    " 6.45059 4.99946 6.66687 5H11V10.5ZM13 8.5H12V5C12 4.73478 11.8946 4.48043 11.7071"
    " 4.29289C11.5196 4.10536 11.2652 4 11 4H6.66687L4.93313 2.7C4.75978 2.57066"
    " 4.54941 2.50054 4.33313 2.5H3.5V1.5H6.33313L8.06688 2.8C8.24022 2.92934 8.45059"
    ' 2.99946 8.66687 3H13V8.5Z" fill="currentColor"/></svg>'
)
REQUEST_ICON = (
    '<svg width="32"  height="32" viewBox="0 0 12 12" fill="none"'
    ' xmlns="http://www.w3.org/2000/svg"><path d="M11 0H1C0.734784 0 0.48043 0.105357'
    " 0.292893 0.292893C0.105357 0.48043 0 0.734784 0 1V11C0 11.2652 0.105357 11.5196"
    " 0.292893 11.7071C0.48043 11.8946 0.734784 12 1 12H11C11.2652 12 11.5196 11.8946"
    " 11.7071 11.7071C11.8946 11.5196 12 11.2652 12 11V1C12 0.734784 11.8946 0.48043"
    " 11.7071 0.292893C11.5196 0.105357 11.2652 0 11 0ZM11 1V7.5H9.20625C9.07499"
    " 7.49966 8.94496 7.5254 8.82372 7.57572C8.70248 7.62604 8.59245 7.69994 8.5"
    " 7.79313L7.29313 9H4.70687L3.5 7.79313C3.40748 7.69986 3.29734 7.62592 3.17599"
    " 7.5756C3.05464 7.52528 2.9245 7.49958 2.79313 7.5H1V1H11ZM11 11H1V8.5H2.79313L4"
    " 9.70687C4.09252 9.80014 4.20266 9.87408 4.32401 9.9244C4.44536 9.97472 4.5755"
    " 10.0004 4.70687 10H7.29313C7.4245 10.0004 7.55464 9.97472 7.67599 9.9244C7.79734"
    ' 9.87408 7.90748 9.80014 8 9.70687L9.20687 8.5H11V11Z" fill="#343330"/></svg>'
)

ARROW_ICON = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="0.5" y="0.5" width="23" height="23" rx="1.5" fill="#ECEBEF"/>'
    '<rect x="0.5" y="0.5" width="23" height="23" rx="1.5" stroke="#B4B0BF"/>'
    '<path d="M17.8538 12.3538L13.3538 16.8538C13.2599 16.9476 13.1327 17.0003 13 17.0003C12.8673 17.0003 12.7401 16.9476 12.6462 16.8538C12.5524 16.76 12.4997 16.6327 12.4997 16.5C12.4997 16.3674 12.5524 16.2401 12.6462 16.1463L16.2931 12.5H6.5C6.36739 12.5 6.24021 12.4474 6.14645 12.3536C6.05268 12.2598 6 12.1326 6 12C6 11.8674 6.05268 11.7402 6.14645 11.6465C6.24021 11.5527 6.36739 11.5 6.5 11.5H16.2931L12.6462 7.85378C12.5524 7.75996 12.4997 7.63272 12.4997 7.50003C12.4997 7.36735 12.5524 7.2401 12.6462 7.14628C12.7401 7.05246 12.8673 6.99976 13 6.99976C13.1327 6.99976 13.2599 7.05246 13.3538 7.14628L17.8538 11.6463C17.9002 11.6927 17.9371 11.7479 17.9623 11.8086C17.9874 11.8693 18.0004 11.9343 18.0004 12C18.0004 12.0657 17.9874 12.1308 17.9623 12.1915C17.9371 12.2522 17.9002 12.3073 17.8538 12.3538Z" fill="#5E5A72"/></svg>'  # noqa: E501
)


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
