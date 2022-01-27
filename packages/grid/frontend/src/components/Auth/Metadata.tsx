import tw from 'twin.macro'
import { Tag } from '$components/Tag'
import { MailLink } from '$components/MailLink'
import { Id } from '$components/Id'

export const AuthDomainMetadata = ({
  domain_name,
  description,
  tags,
  id,
  owner,
  support_email,
}) => (
  <div tw="flex flex-col">
    {tags?.length > 0 && (
      <div tw="flex flex-wrap items-center gap-2 mb-2">
        {tags.map((tag: string) => (
          <Tag color="primary" key={tag} size="small">
            {tag}
          </Tag>
        ))}
      </div>
    )}
    <header>
      <h1>{domain_name}</h1>
      <p tw="mt-6">{description}</p>
    </header>
    <ul tw="flex flex-col gap-4 mt-10 text-sm">
      <li>
        <span tw="text-sm font-bold">ID#:</span> <Id id={id} />
      </li>
      <li>
        <span tw="text-sm font-bold">Owner:</span>{' '}
        <span tw="font-mono text-sm uppercase">{owner}</span>
      </li>
    </ul>
    <hr tw="mt-10" />
    <p tw="mt-8">
      For further assistance please email: <MailLink email={support_email} />
    </p>
  </div>
)
