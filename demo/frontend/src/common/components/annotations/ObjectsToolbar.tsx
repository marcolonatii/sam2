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
import AddObjectButton from '@/common/components/annotations/AddObjectButton';
import FirstClickView from '@/common/components/annotations/FirstClickView';
import ObjectsToolbarBottomActions from '@/common/components/annotations/ObjectsToolbarBottomActions';
import ObjectsToolbarHeader from '@/common/components/annotations/ObjectsToolbarHeader';
// Use non-hook version here as we pass names down
import { getObjectLabel } from '@/common/components/annotations/ObjectUtils';
import ToolbarObject from '@/common/components/annotations/ToolbarObject';
import ObjectNameModal from '@/common/components/annotations/ObjectNameModal'; // Import modal
import {
  activeTrackletObjectAtom,
  activeTrackletObjectIdAtom,
  isAddObjectEnabledAtom,
  isFirstClickMadeAtom,
  trackletObjectsAtom,
  trackletNamesAtom, // Import names atom
  labelTypeAtom, // Import label type atom
} from '@/demo/atoms';
import {BaseTracklet} from '@/common/tracker/Tracker';
import {useAtom, useAtomValue, useSetAtom} from 'jotai';
import {useState, useEffect, useCallback} from 'react'; // Import hooks

type Props = {
  onTabChange: (newIndex: number) => void;
};

// Define state for modal management
type ModalState = {
    isOpen: boolean;
    trackletId: number | null; // ID of tracklet being named/renamed
    currentName: string; // Current name (empty for new, existing for edit)
    isEditing: boolean; // Flag to distinguish initial naming vs editing
};

export default function ObjectsToolbar({onTabChange}: Props) {
  const tracklets = useAtomValue(trackletObjectsAtom);
  const activeTracklet = useAtomValue(activeTrackletObjectAtom);
  const setActiveTrackletId = useSetAtom(activeTrackletObjectIdAtom);
  const isFirstClickMade = useAtomValue(isFirstClickMadeAtom);
  const isAddObjectEnabled = useAtomValue(isAddObjectEnabledAtom);
  const [trackletNames, setTrackletNames] = useAtom(trackletNamesAtom); // Get names state + setter
  const setLabelType = useSetAtom(labelTypeAtom); // Needed after initial object creation

  // State for managing the ObjectNameModal
  const [modalState, setModalState] = useState<ModalState>({
    isOpen: false,
    trackletId: null,
    currentName: '',
    isEditing: false,
  });

  // State to track if the initial naming modal has been triggered
  const [initialNamePrompted, setInitialNamePrompted] = useState(false);

  // Effect to trigger modal for the *first* object after the first click
  useEffect(() => {
      // Only run if the first click was made, we have exactly one tracklet,
      // and we haven't prompted for its name yet.
      if (isFirstClickMade && tracklets.length === 1 && !initialNamePrompted) {
          const firstTracklet = tracklets[0];
          // Check if it *already* somehow has a name (e.g., state persistence)
          const hasName = !!trackletNames[firstTracklet.id];

          if (!hasName && firstTracklet.isInitialized) {
              setModalState({
                  isOpen: true,
                  trackletId: firstTracklet.id,
                  currentName: '', // Start with empty name for the first object
                  isEditing: false, // This is initial naming
              });
              setActiveTrackletId(firstTracklet.id); // Ensure it's active
              setLabelType('positive'); // Set default label type
              setInitialNamePrompted(true); // Mark as prompted
          } else if (hasName || !firstTracklet.isInitialized) {
              // If it has a name or is not initialized yet, mark as prompted anyway
              // to prevent issues if isInitialized flips later without a name prompt.
              setInitialNamePrompted(true);
          }
      }
      // Reset prompted flag if tracklets are cleared
      if (tracklets.length === 0 && initialNamePrompted) {
          setInitialNamePrompted(false);
      }

  }, [isFirstClickMade, tracklets, trackletNames, initialNamePrompted, setActiveTrackletId, setLabelType]);

  // Function to open the modal for editing an existing object's name
  const handleEditName = useCallback((trackletId: number, currentName: string) => {
    setModalState({
        isOpen: true,
        trackletId: trackletId,
        currentName: currentName.startsWith('Object ') ? '' : currentName, // Clear default name placeholder
        isEditing: true, // This is editing
    });
  }, []); // No dependencies needed

  // Function to handle confirmation from the modal
  const handleConfirmName = (newName: string) => {
    if (modalState.trackletId !== null) {
      const finalName = newName.trim();
      setTrackletNames((prev) => {
        const updatedNames = { ...prev };
        if (finalName) {
          updatedNames[modalState.trackletId!] = finalName;
        } else {
          // If the user clears the name, remove the custom name
          // so it reverts to the default "Object X"
          delete updatedNames[modalState.trackletId!];
        }
        return updatedNames;
      });
    }
    // Close the modal and reset state
    setModalState({ isOpen: false, trackletId: null, currentName: '', isEditing: false });
  };

  // Function to handle cancellation from the modal
  const handleCancelName = () => {
    // Close the modal and reset state
    setModalState({ isOpen: false, trackletId: null, currentName: '', isEditing: false });
  };

  if (!isFirstClickMade && tracklets.length === 0) {
    return <FirstClickView />;
  }

  return (
    <>
      <div className="flex flex-col h-full">
        <ObjectsToolbarHeader />
        <div className="grow w-full overflow-y-auto pb-4"> {/* Add padding bottom */}
          {tracklets.map((tracklet: BaseTracklet) => {
            // Use non-hook version inside the map, passing the current names state
            const label = getObjectLabel(tracklet, trackletNames);
            return (
              <ToolbarObject
                key={tracklet.id}
                // Pass label explicitly again, or modify ToolbarObject to accept trackletNames
                tracklet={tracklet}
                isActive={activeTracklet?.id === tracklet.id}
                onClick={() => {
                    // Only set active if not already active, or handle differently if needed
                    if (activeTracklet?.id !== tracklet.id) {
                        setActiveTrackletId(tracklet.id);
                    }
                }}
                onEditName={handleEditName} // Pass the edit handler
              />
            );
          })}
          {isAddObjectEnabled && tracklets.length > 0 && <AddObjectButton />}
        </div>
        {/* Render bottom actions only if there's at least one tracklet */}
        {tracklets.length > 0 && (
            <ObjectsToolbarBottomActions onTabChange={onTabChange} />
        )}
      </div>

      {/* Render the modal conditionally */}
      <ObjectNameModal
        isOpen={modalState.isOpen}
        onConfirm={handleConfirmName}
        onCancel={handleCancelName}
        value={modalState.currentName}
        onChange={(name) => setModalState(prev => ({ ...prev, currentName: name }))}
        modalTitle={modalState.isEditing ? 'Edit object name' : 'Name your object'}
        confirmButtonText={modalState.isEditing ? 'Save Name' : 'Confirm'}
      />
    </>
  );
}