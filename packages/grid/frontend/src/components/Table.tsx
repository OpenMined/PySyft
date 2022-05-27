import React, { useEffect } from 'react'
import cn from 'classnames'
import { ReactNode } from 'react'
import { useTable, useRowSelect, useSortBy, useGlobalFilter } from 'react-table'
import { Checkbox, Text } from '@/omui'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSortDown, faSortUp } from '@fortawesome/free-solid-svg-icons'

interface TableContentProps {
  className?: string
  isLast?: boolean
  sortable?: boolean
  isSorted?: boolean
  isSortedDesc?: boolean
  children: ReactNode
}

export function TableHeader({
  className,
  children,
  isLast,
  sortable,
  isSorted,
  isSortedDesc,
}: TableContentProps) {
  return (
    <div
      className={cn(
        'w-full flex justify-between items-center px-2 h-12 border-b',
        isSorted && 'bg-gray-50',
        !isLast && 'border-r'
      )}
    >
      <div
        className={cn('space-x-2 flex items-center justify-center', className)}
      >
        {['string', 'number'].includes(typeof children) && (
          <Text size="sm">{children}</Text>
        )}
        {typeof children === 'object' && children}
      </div>
      {isSorted && (
        <FontAwesomeIcon
          icon={isSortedDesc ? faSortUp : faSortDown}
          className="pl-2 flex-shrink-0"
        />
      )}
    </div>
  )
}

export function TableItemOuter({
  className,
  children,
  center,
  isLast,
}: {
  className?: string
  center?: boolean
  isLast?: boolean
  children: ReactNode
}) {
  return (
    <div
      className={cn(
        'space-x-2 px-4 h-12 flex items-center border-b w-full',
        !isLast && 'border-r',
        center && 'justify-center',
        className
      )}
    >
      {children}
      {/* {['string', 'number'].includes(typeof children) && <Text size="sm">{children}</Text>} */}
      {/* {typeof children === 'object' && children} */}
    </div>
  )
}

const TableCheckbox = React.forwardRef((props, ref) => {
  const defaultRef = React.useRef()
  const resolvedRef = ref || defaultRef

  return <Checkbox ref={resolvedRef} {...props} />
})

export function TableItem({
  className,
  center,
  children,
}: {
  className?: string
  center?: boolean
  children: ReactNode
}) {
  return (
    <div className={cn('w-full', center && 'text-center', className)}>
      {children}
    </div>
  )
}

export function useOMUITable({ data, columns, selectable, sortable }) {
  const tableInstance = useTable(
    { columns, data },
    useGlobalFilter,
    useSortBy,
    useRowSelect
  )

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
    selectedFlatRows,
    state: { selectedRowIds },
  } = tableInstance

  return {
    instance: tableInstance,
    Component: (
      <div className="min-w-full">
        <table {...getTableProps()} className="w-full border-t">
          <thead>
            {headerGroups.map((headerGroup) => (
              <tr {...headerGroup.getHeaderGroupProps()}>
                {headerGroup.headers.map((column, index) => (
                  <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                    <TableHeader
                      sortable={sortable}
                      isLast={index + 1 === headerGroup.headers.length}
                      isSorted={column.isSorted}
                      isSortedDesc={column.isSortedDesc}
                    >
                      {column.render('Header')}
                    </TableHeader>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody {...getTableBodyProps()}>
            {rows.map((row) => {
              prepareRow(row)
              return (
                <tr {...row.getRowProps()}>
                  {row.cells.map((cell, index) => {
                    return (
                      <td
                        {...cell.getCellProps()}
                        className={cn(
                          cell.column.isSorted && 'bg-gray-50',
                          row.isSelected && 'bg-primary-50'
                        )}
                      >
                        {selectable && index === 0 ? (
                          <div className="flex h-12 space-x-1 w-full items-center border-b border-r">
                            <div className="w-10 flex justify-center">
                              <TableCheckbox
                                {...row.getToggleRowSelectedProps()}
                              />
                            </div>
                            <div className="w-full">{cell.render('Cell')}</div>
                          </div>
                        ) : (
                          <TableItemOuter
                            isLast={index + 1 === row.cells.length}
                          >
                            {cell.render('Cell')}
                          </TableItemOuter>
                        )}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    ),
  }
}
