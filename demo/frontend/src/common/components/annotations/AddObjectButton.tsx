/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar';
import useVideo from '@/common/components/video/editor/useVideo';
import {activeTrackletObjectIdAtom, labelTypeAtom, trackletNamesAtom} from '@/demo/atoms';
import {Add} from '@carbon/icons-react';
import {useAtom} from 'jotai';
import {useState} from 'react';
import ObjectNameModal from '@/common/components/annotations/ObjectNameModal';

export default function AddObjectButton() {
  const video = useVideo();
  const [trackletNames, setTrackletNames] = useAtom(trackletNamesAtom);
  const [activeTrackletId, setActiveTrackletId] = useAtom(activeTrackletObjectIdAtom);
  const setLabelType = useAtom(labelTypeAtom)[1];
  const {enqueueMessage} = useMessagesSnackbar();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [objectName, setObjectName] = useState('');

  function handleOpenModal() {
    setObjectName(''); // Reset name field when opening for a new object
    setIsModalOpen(true);
  }

  async function handleConfirmName(name: string) {
    setIsModalOpen(false); // Close modal first
    enqueueMessage('addObjectClick');
    const tracklet = await video?.createTracklet();
    if (tracklet != null) {
      setActiveTrackletId(tracklet.id);
      setLabelType('positive');
      const finalName = name.trim();
      if (finalName) {
        // Use functional update for atom
        setTrackletNames((prev) => ({ ...prev, [tracklet.id]: finalName }));
      }
      // If name is empty, default name "Object X" will be used by getObjectLabel
    }
    // Reset object name state only after processing
    setObjectName('');
  }

  function handleCancel() {
    setIsModalOpen(false);
    setObjectName('');
  }

  return (
    <>
      <div
        onClick={handleOpenModal}
        className="group flex justify-start mx-4 px-4 py-4 bg-transparent text-white !rounded-xl border-none cursor-pointer hover:bg-graydark-800/50 transition-colors duration-150" // Added padding and hover effect
        role="button" // Semantics
        tabIndex={0} // Make it focusable
        onKeyDown={(e) => e.key === 'Enter' && handleOpenModal()} // Keyboard accessibility
      >
        <div className="flex gap-6 items-center">
          <div className=" group-hover:bg-graydark-700 border border-white relative h-12 w-12 md:w-20 md:h-20 shrink-0 rounded-lg flex items-center justify-center transition-colors duration-150">
            <Add size={36} className="group-hover:text-white text-gray-300 transition-colors duration-150" />
          </div>
          <div className="font-medium text-base">Add another object</div>
        </div>
      </div>
      <ObjectNameModal
        isOpen={isModalOpen}
        onConfirm={handleConfirmName}
        onCancel={handleCancel}
        value={objectName}
        onChange={setObjectName}
        // Explicitly set props for adding a new object
        modalTitle="Name your new object"
        confirmButtonText="Create Object"
      />
    </>
  );
}