#include "iostream"
namespace pdb{
    class Replacer {

    public:

        Replacer() = default;

        ~Replacer() = default;
        /**
         * Remove the victim frame as defined by the replacement policy.
         * @param[out] frame_id id of frame that was removed, nullptr if no victim was found
         * @return true if a victim frame was found, false otherwise
         */
        bool Victim(frame_id_t *frame_id);

        /**
         * Pins a frame, indicating that it should not be victimized until it is unpinned.
         * @param frame_id the id of the frame to pin
         */

        void Pin(frame_id_t frame_id);

        /**
         * Unpins a frame, indicating that it can now be victimized.
         * @param frame_id the id of the frame to unpin
         */
        void Unpin(frame_id_t frame_id);

        /** @return the number of elements in the replacer that can be victimized */
        size_t Size();
    };
};
