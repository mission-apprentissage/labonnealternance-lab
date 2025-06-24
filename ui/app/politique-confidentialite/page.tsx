import { Suspense } from "react";

import Breadcrumb from "@/components/breadcrumb/Breadcrumb";

import { PAGES } from "../../utils/routes.utils";
import PolitiqueConfidentialite from "./components/PolitiqueConfidentialite";

const PolitiqueConfidentialitePage = () => {
  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.politiqueConfidentialite]} />
      <PolitiqueConfidentialite />
    </Suspense>
  );
};
export default PolitiqueConfidentialitePage;
