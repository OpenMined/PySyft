import tw, { styled } from 'twin.macro'
import { Logo } from '$components/Logo'
import { TagList } from '$components/TagList'
import { Ellipses } from '$components/Ellipses'
import { Id } from '$components/Id'
import { MailLink } from '$components/MailLink'

const GridContainer = styled.section`
  ${tw`grid grid-cols-12 gap-4 h-full`}

  grid-template-rows: 80px auto 88px;

  > * {
    ${tw`col-span-10 col-start-2`}
  }

  footer {
    ${tw`flex items-center gap-2 w-full mt-6`}
  }

  footer img {
    ${tw`max-h-6`}
  }

  header > img {
    ${tw`max-h-[80px] object-contain`}
  }
`

export const AuthLayout = ({ children }) => {
  const {
    tags = ['Commodities', 'Trade', 'Canada'],
    domain_name = 'Canada Domain',
    owner = 'Kyoko Eng',
    description = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis in vulputate enim. Morbi scelerisque, ante eu ultrices semper, ipsum nisl malesuada ligula, non faucibus libero purus et ligula.',
    id = '449f4f997a96467f90f7af8b396928f1	',
    support_email = 'support@abc.com',
  } = {}

  return (
    <div tw="h-full w-full flex justify-center">
      <main tw="relative max-h-full max-w-[1440px]">
        <GridContainer tw="z-10">
          <Header />
          <section tw="(col-start-2 col-span-4)!">
            <div tw="flex flex-col">
              {tags?.length > 0 && (
                <div tw="flex flex-wrap items-center gap-2 mb-2">
                  <TagList tags={tags} />
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
          </section>
          <section tw="(col-span-5 col-start-7)!">{children}</section>
          <Footer />
        </GridContainer>
        <div tw="w-3/5 absolute right-0 top-[-13px] z-[-1]">
          <Ellipses />
        </div>
      </main>
    </div>
  )
}

const Header = () => (
  <header>
    <Logo lockup="horizontal" product="pygrid" color="light" />
  </header>
)

const Footer = () => (
  <footer>
    <p tw="text-xs">Empowered by</p>
    <a href="https://openmined.org" target="_blank" rel="noopener noreferrer">
      <Logo lockup="horizontal" product="org" color="light" />
    </a>
  </footer>
)
