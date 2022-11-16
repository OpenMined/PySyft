import { Highlight } from '@/components'

const datasetFileRequirements: DatasetRequirementsList = [
  {
    term: 'tags',
    description:
      'Contains a list of tags, one tag per line, written in the format #tag',
  },
  {
    term: 'description',
    description:
      'This is a description of the dataset, containing all information about what this dataset represents. Variable definitions, how data was collected, what units were used, if this was part of a research, the objective in collecting this data, amongst many other interesting informations about the dataset. Avoid addressing the data itself — present the data in the manifest file (see below).',
  },
  {
    term: 'manifest',
    description:
      "The manifest file contains a list of all files included in the dataset, including basic file info (e.g. file name) along with extended file metadata. Other useful information often found in manifest files are the name of columns, format of column values, number of records. Use this file for general information on the data itself — it's a description of the data.",
  },
]

type DatasetRequirementsList = { term: string; description: string }[]

function Description({ children }) {
  return <dd>{children}</dd>
}

function Term({ children }) {
  return <dt className="italic font-medium">{children}</dt>
}

export function DatasetRequirementsList() {
  return (
    <>
      <h3 className="font-medium">Requirements</h3>
      <p>
        PyGrid currently accepts tarball files <Highlight>.tar.gz</Highlight>.
        It is advised to add the following files to your archive:{' '}
        <Highlight>tags</Highlight>, <Highlight>description</Highlight> and{' '}
        <Highlight>manifest</Highlight>. These files should be saved without any
        extensions and only contain plain text. The following is a quick
        description of each file:
      </p>
      <dl className="space-y-3">
        {datasetFileRequirements.map(({ term, description }) => (
          <div key={term}>
            <Term>{term}</Term>
            <Description>{description}</Description>
          </div>
        ))}
      </dl>
      <h3 className="font-medium">Permissions</h3>
      <p>
        All users with{' '}
        <Highlight className="uppercase">Can upload data</Highlight> are allowed
        to create new datasets in the domain.
      </p>
    </>
  )
}
