import cn from 'classnames'
import type {ReactNode} from 'react'

function TableHeader({children}) {
  return (
    <th scope="col" className="px-5 py-3 text-xs uppercase tracking-wide text-gray-500">
      {children}
    </th>
  )
}

export function TableData({className, children}: {className?: string; children: ReactNode}) {
  return <td className={cn('px-5 py-3 text-xs whitespace-nowrap text-gray-500', className)}>{children}</td>
}

export function TableRow({darkBackground, children}) {
  return <tr className={darkBackground ? 'bg-trueGray-200 bg-opacity-40' : 'bg-white'}>{children}</tr>
}

interface Table {
  headers: string[]
  children: ReactNode
}

export function Table({headers, children}: Table) {
  return (
    <div className="shadow overflow-hidden border-b border-trueGray-200 rounded-md ">
      <table cellSpacing={4} cellPadding={4} className="min-w-full divide-y divide-trueGray-200">
        <thead className="bg-trueGray-200">
          <tr className="text-left">
            {headers.map(columnHeader => (
              <TableHeader key={columnHeader}>{columnHeader}</TableHeader>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}
