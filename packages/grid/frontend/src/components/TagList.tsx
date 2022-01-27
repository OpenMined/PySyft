import { Tag } from '$components/Tag'

interface TagListProps {
  tags: Array<string>
}

export const TagList = ({ tags = [] }: TagListProps) => (
  <>
    {tags.map((tag: string) => (
      <Tag color="primary" key={tag} size="small">
        {tag}
      </Tag>
    ))}
  </>
)
