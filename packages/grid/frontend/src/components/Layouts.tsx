import tw, { css, styled } from 'twin.macro'
import { SideMenu } from '$components/SideMenu'

const MainContent = styled.div`
  ${tw`grid grid-cols-12 gap-6 pl-10 max-w-[1170px] mr-auto w-full h-full`}
  ${tw`rounded-tl-[15px]`}
  ${tw`py-8`}
`

const DesktopLayout = styled.div`
  display: grid;
  grid-template-areas: 'side-menu main-content';
  grid-template-columns: 270px 1fr;
  ${tw`bg-gray-800 bg-scrim-gray-dark`}
`

export const SimpleCenter = ({ children }) => (
  <DesktopLayout>
    <SideMenu />
    <div tw="bg-[#F1F0F4] bg-scrim-layout-white">
      <MainContent>{children}</MainContent>
    </div>
  </DesktopLayout>
)
