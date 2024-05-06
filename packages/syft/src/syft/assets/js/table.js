document.querySelectorAll(".escape-unfocus").forEach((input) => {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      console.log("Escape key pressed");
      event.stopPropagation();
      input.blur();
    }
  });
});

function buildTable(columns, rowHeader, data, uid) {
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
    layout: "fitDataStretch",
    resizableColumnFit: true,
    resizableColumnGuide: true,
    pagination: "local",
    paginationSize: 5,
    height: "500px",
  });

  // Fix cell height issue after switching pages
  table.on("pageLoaded", function (_pageno) {
    table.redraw();
  });
  // Redraw on resize: cell hight gets recalcualted for overflowing content
  table.on("columnResized", function (_column) {
    table.redraw();
  });

  // set number of rows
  const numrowsElement = document.querySelector(`#${numrowsId}`);
  if (numrowsElement) {
    numrowsElement.innerHTML = data.length;
  }

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
