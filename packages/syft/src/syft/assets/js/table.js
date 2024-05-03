document.querySelectorAll(".escape-unfocus").forEach((input) => {
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      console.log("Escape key pressed");
      event.stopPropagation();
      input.blur();
    }
  });
});

function buildTable(columns, data, tableId, searchBarId) {
  const tableElement = document.querySelector(`#${tableId}`);
  if (!tableElement) {
    console.error(`Element with id "${tableId}" not found.`);
    return;
  }

  const table = new Tabulator(`#${tableId}`, {
    data: data,
    columns: columns,
    layout: "fitColumns",
    resizableColumnFit: true,
    resizableColumnGuide: true,
    pagination: "local",
    paginationSize: 5,
    height: "500px",
  });

  // TODO dark theme
  // var body = document.querySelector('body');
  // if (body.getAttribute('data-jp-theme-light') === 'false') {
  //     tableElement.querySelector('.tabulator-table').classList.add('table-dark');
  // } else {
  //     tableElement.querySelector('.tabulator-table').classList.remove('table-dark');
  // }

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
