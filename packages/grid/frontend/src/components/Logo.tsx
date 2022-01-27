import tw, { styled } from 'twin.macro'

type Horizontal = { lockup: 'horizontal'; product: 'org' | 'pygrid'; color: 'light' }
type MarkOrg = { lockup: 'mark'; product: 'org'; color: 'light' | 'dark' | 'black' }
type MarkPyGrid = { lockup: 'mark'; product: 'pygrid'; color: 'light' }

type LogoProps = Horizontal | MarkOrg | MarkPyGrid

const Img = styled.img`
  ${tw`object-contain`}
`

export const Logo = ({ lockup, color, product }: LogoProps) => {
  const name = product === 'org' ? 'OpenMined logo' : 'PyGrid logo'
  return <Img src={`/assets/logo-${product}-${lockup}-${color}.svg`} alt={name} />
}
