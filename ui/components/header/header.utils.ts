import type { MainNavigationProps } from "@codegouvfr/react-dsfr/MainNavigation";
import { useMemo } from "react";
import type { IUserPublic } from "shared/src/models/user.model";

import { PAGES } from "@/utils/routes.utils";

type GetNavigationItemsProps = {
  user: IUserPublic | null;
  pathname: string;
};

const getNavigationItems = ({ user, pathname }: GetNavigationItemsProps): MainNavigationProps.Item[] => {
  const navigation: MainNavigationProps.Item[] = [
    {
      isActive: pathname === PAGES.static.home.getPath(),
      text: PAGES.static.home.getTitle(),
      linkProps: {
        href: PAGES.static.home.getPath(),
      },
    },
    {
      isActive: pathname.startsWith(PAGES.static.explorerApi.getPath()),
      text: PAGES.static.explorerApi.getTitle(),
      linkProps: {
        href: PAGES.static.explorerApi.getPath(),
      },
    },
  ];

  if (user?.is_admin) {
    const adminMenuLinks = [
      {
        text: PAGES.static.adminOrganisations.getTitle(),
        isActive: pathname.startsWith(PAGES.static.adminOrganisations.getPath()),
        linkProps: {
          href: PAGES.static.adminOrganisations.getPath(),
        },
      },
      {
        text: PAGES.static.adminUsers.getTitle(),
        isActive: pathname.startsWith(PAGES.static.adminUsers.getPath()),
        linkProps: {
          href: PAGES.static.adminUsers.getPath(),
        },
      },
      {
        text: PAGES.static.adminProcessor.getTitle(),
        isActive: pathname.startsWith(PAGES.static.adminProcessor.getPath()),
        linkProps: {
          href: PAGES.static.adminProcessor.getPath(),
        },
      },
    ];

    navigation.push({
      text: "Administration",
      isActive: adminMenuLinks.some((link) => link.isActive),
      menuLinks: adminMenuLinks,
    });
  }

  return navigation.map((item) => {
    const { menuLinks } = item;

    const menuLinkWithActive = menuLinks?.map((link) => ({ ...link, isActive: link.linkProps.href === pathname }));
    const isActive = item.isActive || menuLinkWithActive?.some((link) => link.isActive);

    return { ...item, isActive, menuLinks };
  }) as MainNavigationProps.Item[];
};

export const useNavigationItems = ({ user, pathname }: GetNavigationItemsProps): MainNavigationProps.Item[] =>
  useMemo(() => getNavigationItems({ user, pathname }), [user, pathname]);
