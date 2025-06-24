import { Breadcrumb as DSFRBreadcrumb } from "@codegouvfr/react-dsfr/Breadcrumb";

import type { IPage } from "@/utils/routes.utils";
import { PAGES } from "@/utils/routes.utils";

export default function Breadcrumb({ pages }: { pages: IPage[] }) {
  const rest = [...pages];
  const currentPage = rest.pop();

  return (
    <DSFRBreadcrumb
      currentPageLabel={currentPage?.getTitle()}
      homeLinkProps={{
        href: PAGES.static.home.getPath(),
      }}
      segments={rest.map((page) => ({
        label: page.getTitle(),
        linkProps: {
          href: page.getPath(),
        },
      }))}
    />
  );
}
