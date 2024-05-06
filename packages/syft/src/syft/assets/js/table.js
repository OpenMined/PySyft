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
