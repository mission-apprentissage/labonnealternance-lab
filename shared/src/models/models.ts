import type { z } from "zod";

import type { IModelDescriptorGeneric } from "./common";
import emailDeniedModel from "./email_denied.model";
import emailEventModel from "./email_event.model";
import { organisationModelDescriptor } from "./organisation.model";
import sessionModel from "./session.model";
import userModel from "./user.model";

export const modelDescriptorMap = {
  [sessionModel.collectionName]: sessionModel,
  [userModel.collectionName]: userModel,
  [emailEventModel.collectionName]: emailEventModel,
  [emailDeniedModel.collectionName]: emailDeniedModel,
  [organisationModelDescriptor.collectionName]: organisationModelDescriptor,
};

export type IModelDescriptorMap = typeof modelDescriptorMap;

export type IModelDescriptor = IModelDescriptorMap[keyof IModelDescriptorMap];

export const modelDescriptors = Object.values(
  modelDescriptorMap
) as IModelDescriptorMap[keyof IModelDescriptorMap][] satisfies IModelDescriptorGeneric[];

export type CollectionName = keyof typeof modelDescriptorMap;

export type IDocument<Name extends CollectionName> = z.output<(typeof modelDescriptorMap)[Name]["zod"]>;
