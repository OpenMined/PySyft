TABULATOR_SRC =
  "https://unpkg.com/tabulator-tables@6.2.1/dist/js/tabulator.min";
TABULATOR_CSS =
  "https://cdn.jsdelivr.net/gh/openmined/pysyft/packages/syft/src/syft/assets/css/tabulator_pysyft.min.css";

document.querySelectorAll(".escape-unfocus").forEach((input) => {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.stopPropagation();
      input.blur();
    }
  });
});

function load_script(scriptPath, elementId, on_load, on_error) {
  var element = document.getElementById(elementId);
  var script = document.createElement("script");
  script.type = "application/javascript";
  script.src = scriptPath;
  script.onload = on_load;
  script.onerror = on_error;
  console.debug("Injecting script:", scriptPath);
  element.appendChild(script);
}

function load_css(cssPath, elementId, on_load, on_error) {
  var element = document.getElementById(elementId);
  var css = document.createElement("link");
  css.onload = on_load;
  css.onerror = on_error;
  css.rel = "stylesheet";
  css.type = "text/css";
  css.href = cssPath;
  console.debug("Injecting css:", cssPath);
  element.appendChild(css);
}

function fix_url_for_require(url) {
  return url.endsWith(".js") ? url.replace(/(\.js)(?!.*\1)/, "") : url;
}

function load_tabulator(elementId) {
  load_css(TABULATOR_CSS, elementId);

  return new Promise((resolve, reject) => {
    if (typeof require !== "undefined") {
      url = fix_url_for_require(TABULATOR_SRC);
      return require([url], function (module) {
        window.Tabulator = module;
        resolve();
      }, reject);
    } else if (typeof window.Tabulator === "undefined") {
      load_script(TABULATOR_SRC, elementId, resolve, reject);
    } else {
      resolve();
    }
  });
}

function buildTable(
  columns,
  rowHeader,
  data,
  uid,
  pagination = true,
  maxHeight = null,
  headerSort = true,
) {
  const tableId = `table-${uid}`;
  const searchBarId = `search-${uid}`;
  const numrowsId = `numrows-${uid}`;

  const tableElement = document.querySelector(`#${tableId}`);
  if (!tableElement) {
    console.error(`Element with id "${tableId}" not found.`);
    return;
  }

  const table = new Tabulator(`#${tableId}`, {
    data: data,
    columns: columns,
    rowHeader: rowHeader,
    index: "_table_repr_index",
    layout: "fitDataStretch",
    resizableColumnFit: true,
    resizableColumnGuide: true,
    pagination: pagination,
    paginationSize: 5,
    maxHeight: maxHeight,
    headerSort: headerSort,
  });

  // Events needed for cell overflow:
  // fixes incomplete border + cells too much height for overflowing cells
  table.on("pageLoaded", function (_pageno) {
    // paginate
    table.redraw();
  });
  table.on("columnResized", function (_column) {
    // drag resize
    table.redraw();
  });
  table.on("tableBuilt", function () {
    // first build
    table.redraw();
  });

  const numrowsElement = document.querySelector(`#${numrowsId}`);
  if (numrowsElement) {
    numrowsElement.innerHTML = data.length;
  }

  configureHighlightSingleRow(table, uid);
  configureSearch(table, searchBarId, columns);

  return table;
}

function configureSearch(table, searchBarId, columns) {
  // https://stackoverflow.com/questions/76208880/tabulator-global-search-across-multiple-columns
  const searchBar = document.getElementById(searchBarId);
  if (!searchBar) {
    console.error(`Element with id "${searchBarId}" not found.`);
    return;
  }

  const columnFields = columns.map((column) => column.field);
  const ignoreColumns = [];
  const searchFields = columnFields.filter(
    (field) => !ignoreColumns.includes(field),
  );

  searchBar.addEventListener("input", function () {
    let searchValue = searchBar.value.trim();

    let filterArray = searchFields.map((field) => {
      return { field: field, type: "like", value: searchValue };
    });

    table.setFilter([filterArray]);
  });
}

function configureHighlightSingleRow(table, uid) {
  // Listener for rowHighlight events, with fields:
  //    uid: string, table uid
  //    index: number | string, row index to highlight
  //    jumpToRow: bool, if true, jumps to page where the row is located
  document.addEventListener("rowHighlight", function (e) {
    if (e.detail.uid === uid) {
      let row_idx = e.detail.index;
      let rows = table.getRows();
      for (let row of rows) {
        if (row.getIndex() == row_idx) {
          row.select();
          if (e.detail.jumpToRow) {
            // catch promise in case the table does not have pagination
            table.setPageToRow(row_idx).catch((_) => {});
            table.scrollToRow(row_idx, "top", false);
          }
        } else {
          row.deselect();
        }
      }
    }
  });
}

function waitForTable(uid, timeout = 1000) {
  return new Promise((resolve, reject) => {
    // Check if the table is ready immediately
    if (window["table_" + uid]) {
      resolve();
    } else {
      // Otherwise, poll until the table is ready or timeout
      var startTime = Date.now();
      var checkTableInterval = setInterval(function () {
        if (window["table_" + uid]) {
          clearInterval(checkTableInterval);
          resolve();
        } else if (Date.now() - startTime > timeout) {
          clearInterval(checkTableInterval);
          reject(`Timeout: table_"${uid}" not found.`);
        }
      }, 100);
    }
  });
}

function highlightSingleRow(uid, index = null, jumpToRow = false) {
  // Highlight a single row in the table with the given uid
  // If index is not provided or doesn't exist, all rows are deselected
  waitForTable(uid)
    .then(() => {
      document.dispatchEvent(
        new CustomEvent("rowHighlight", {
          detail: { uid, index, jumpToRow },
        }),
      );
    })
    .catch((error) => {
      console.log(error);
    });
}

function updateTableCell(uid, index, field, value) {
  // Update the value of a cell in the table with the given uid
  waitForTable(uid)
    .then(() => {
      const table = window["table_" + uid];
      if (!table) {
        throw new Error(`Table with uid ${uid} not found.`);
      }

      const row = table.getRow(index);
      if (!row) {
        throw new Error(`Row with index ${index} not found.`);
      }

      // Update the cell value
      row.update({ [field]: value });
    })
    .catch((error) => {
      console.error(error);
    });
}
