import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import Accessibilite from "./components/Accessibilite";

const AccessibilitePage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.accessibilite]} />
      <Accessibilite />
    </Suspense>
  );
};
export default AccessibilitePage;
